"""Phase 5 — Real Rating Score (RRS) aggregation.

    uv run python -m rrs.scoring            # → models/rrs_scores.parquet + reports/rrs_<metro>.md

Turns per-review `p_fake` (Phase 4, `models/predictions.parquet`) into a per-business
adjusted rating. Each review is weighted by its *trust* (`1 - p_fake`), so a review the
model thinks is likely fake counts for less but is never silently dropped. The
trust-weighted average is then Bayesian-shrunk toward the global mean so businesses with
few (or few trustworthy) reviews don't swing wildly.

The output is deliberately a *bundle*, never a bare number — RRS, a confidence interval,
and transparency counts (how many reviews were flagged) — because `p_fake` is a noisy
second opinion, not a verdict (see the plan's "Honest limitations").

Per business, with trust weight wᵢ = 1 − p_fakeᵢ:

    weighted_sum = Σ starsᵢ · wᵢ
    weight_total = Σ wᵢ                      ( = effective sample size n_eff )
    rrs          = (weighted_sum + K · μ) / (weight_total + K)     K = PRIOR_WEIGHT

μ is the global trust-weighted mean rating. The Wilson interval is computed on the rating
rescaled to [0, 1] (`(r − 1) / 4`) using n_eff, then mapped back to the 1–5 scale.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import polars as pl

from rrs.config import DEFAULT_METRO, REPORTS_DIR, ROOT

PREDICTIONS_PATH = ROOT / "models" / "predictions.parquet"
FEATURES_PATH = ROOT / "features" / "reviews.parquet"
OUT_PATH = ROOT / "models" / "rrs_scores.parquet"

# Bayesian shrinkage strength: with K=10 a business needs ~10 trustworthy reviews before
# its own ratings outweigh the global prior. Tunable; matches the plan.
PRIOR_WEIGHT = 10.0

# A review is "flagged" (counted in the transparency stats) at p_fake above this. The RRS
# itself uses the continuous weight, not this threshold.
FLAG_THRESHOLD = 0.5

# Rating scale bounds (Yelp stars).
STARS_MIN, STARS_MAX = 1.0, 5.0

Z_95 = 1.959963984540054  # standard normal quantile for a 95% two-sided interval


def wilson_interval(p_hat: float, n: float, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a proportion `p_hat` observed over effective size `n`.

    Robust at small/effective-fractional n (unlike the normal approximation), which is the
    whole reason we use it for thin-evidence businesses. Returns (low, high) clamped to
    [0, 1]. n ≤ 0 collapses to the point estimate."""
    if n <= 0:
        return (p_hat, p_hat)
    p_hat = min(max(p_hat, 0.0), 1.0)
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p_hat * (1 - p_hat) / n + z2 / (4 * n * n))
    return (max(0.0, center - margin), min(1.0, center + margin))


def _rating_ci(rrs: float, n_eff: float) -> tuple[float, float]:
    """Wilson CI on the rating, via the [0,1]-rescaled rating and back to the 1–5 scale."""
    span = STARS_MAX - STARS_MIN
    p_hat = (rrs - STARS_MIN) / span
    lo, hi = wilson_interval(p_hat, n_eff)
    return (STARS_MIN + lo * span, STARS_MIN + hi * span)


def global_trust_weighted_mean(reviews: pl.DataFrame) -> float:
    """μ — the global mean rating weighted by trust (1 − p_fake). This is the 'authentic'
    baseline the per-business score shrinks toward."""
    num = (reviews["stars"] * (1.0 - reviews["p_fake"])).sum()
    den = (1.0 - reviews["p_fake"]).sum()
    return float(num / den) if den > 0 else float(reviews["stars"].mean())


def score_businesses(reviews: pl.DataFrame, prior_weight: float = PRIOR_WEIGHT) -> pl.DataFrame:
    """Aggregate a (business_id, stars, p_fake) frame into per-business RRS + transparency.

    Returns one row per business with: n_reviews, naive_mean (the unweighted star average,
    i.e. Yelp's number), rrs, rrs_ci_low/high, n_eff, pct_flagged, n_flagged,
    n_authentic_reviews, and the global mean used."""
    mu = global_trust_weighted_mean(reviews)

    per_biz = (
        reviews.with_columns(
            (1.0 - pl.col("p_fake")).alias("trust"),
            (pl.col("p_fake") > FLAG_THRESHOLD).alias("flagged"),
        )
        .group_by("business_id")
        .agg(
            pl.len().alias("n_reviews"),
            pl.col("stars").mean().alias("naive_mean"),
            (pl.col("stars") * pl.col("trust")).sum().alias("weighted_sum"),
            pl.col("trust").sum().alias("weight_total"),
            pl.col("flagged").sum().alias("n_flagged"),
        )
        .with_columns(
            ((pl.col("weighted_sum") + prior_weight * mu) / (pl.col("weight_total") + prior_weight))
            .alias("rrs"),
            pl.col("weight_total").alias("n_eff"),
            (pl.col("n_flagged") / pl.col("n_reviews")).alias("pct_flagged"),
            (pl.col("n_reviews") - pl.col("n_flagged")).alias("n_authentic_reviews"),
            pl.lit(mu).alias("global_mean"),
        )
    )

    # Wilson CI is per-row scalar work; map it in Python (44.8K rows — trivially fast).
    ci = [_rating_ci(r, n) for r, n in zip(per_biz["rrs"], per_biz["n_eff"], strict=True)]
    per_biz = per_biz.with_columns(
        pl.Series("rrs_ci_low", [c[0] for c in ci]),
        pl.Series("rrs_ci_high", [c[1] for c in ci]),
        (pl.col("rrs") - pl.col("naive_mean")).alias("rrs_delta"),
    )
    return per_biz.select(
        "business_id", "n_reviews", "naive_mean", "rrs", "rrs_delta",
        "rrs_ci_low", "rrs_ci_high", "n_eff", "pct_flagged", "n_flagged",
        "n_authentic_reviews", "global_mean",
    ).sort("business_id")


def load_review_scores(
    predictions_path: Path = PREDICTIONS_PATH,
    features_path: Path = FEATURES_PATH,
) -> pl.DataFrame:
    """review_id → (business_id, stars, p_fake) by joining Phase 4 predictions to the
    feature table (which carries business_id + stars)."""
    preds = pl.read_parquet(predictions_path, columns=["review_id", "p_fake"])
    feats = pl.read_parquet(features_path, columns=["review_id", "business_id", "stars"])
    joined = feats.join(preds, on="review_id", how="inner")
    if joined.height != preds.height:
        raise ValueError(
            f"prediction/feature row mismatch: {preds.height:,} predictions, "
            f"{joined.height:,} joined. Rebuild Phase 3/4 outputs."
        )
    return joined


def _write_report(scores: pl.DataFrame, path: Path, min_reviews: int = 20) -> None:
    """Distribution of RRS vs the naive mean + the businesses whose rating moves most."""
    mu = float(scores["global_mean"][0])
    lines: list[str] = []
    w = lines.append
    w(f"# Real Rating Score — {DEFAULT_METRO}\n")
    w(f"Per-business RRS over {scores.height:,} businesses. Global trust-weighted mean "
      f"μ = {mu:.3f}, prior weight K = {PRIOR_WEIGHT:.0f}.\n")

    w("## Distribution\n")
    w("| stat | naive mean | RRS | RRS − naive |")
    w("|---|---|---|---|")
    for q, name in [(0.1, "p10"), (0.25, "p25"), (0.5, "p50"), (0.75, "p75"), (0.9, "p90")]:
        w(f"| {name} | {scores['naive_mean'].quantile(q):.3f} | "
          f"{scores['rrs'].quantile(q):.3f} | {scores['rrs_delta'].quantile(q):+.3f} |")
    w(f"| mean | {scores['naive_mean'].mean():.3f} | {scores['rrs'].mean():.3f} | "
      f"{scores['rrs_delta'].mean():+.3f} |")
    w("")
    w(f"Mean |RRS − naive| = {scores['rrs_delta'].abs().mean():.3f} stars. "
      f"Median % of reviews flagged per business = {scores['pct_flagged'].median():.3f}.\n")

    eligible = scores.filter(pl.col("n_reviews") >= min_reviews)
    w(f"## Largest downward adjustments (≥{min_reviews} reviews)\n")
    w("Businesses whose RRS sits most below their Yelp mean — the most 'rating-inflated'.\n")
    w("| business_id | n | naive | RRS | Δ | % flagged |")
    w("|---|---|---|---|---|---|")
    for r in eligible.sort("rrs_delta").head(15).iter_rows(named=True):
        w(f"| {r['business_id']} | {r['n_reviews']} | {r['naive_mean']:.2f} | "
          f"{r['rrs']:.2f} | {r['rrs_delta']:+.2f} | {r['pct_flagged']:.2f} |")
    w("")
    w("> RRS is a second opinion, not a verdict. A downward adjustment means the model "
      "trusted a business's reviews less on average — often a low-activity reviewer base — "
      "not proof of manipulation.\n")

    path.write_text("\n".join(lines))
    print(f"  wrote {path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 5 — Real Rating Score aggregation.")
    ap.add_argument("--predictions", type=Path, default=PREDICTIONS_PATH)
    ap.add_argument("--features", type=Path, default=FEATURES_PATH)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    ap.add_argument("--prior-weight", type=float, default=PRIOR_WEIGHT)
    args = ap.parse_args()

    print("→ load review-level p_fake + stars", flush=True)
    reviews = load_review_scores(args.predictions, args.features)
    print(f"  {reviews.height:,} reviews", flush=True)

    print("→ aggregate per business", flush=True)
    scores = score_businesses(reviews, prior_weight=args.prior_weight)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    scores.write_parquet(args.out)
    print(f"  wrote {args.out} — {scores.height:,} businesses", flush=True)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    _write_report(scores, REPORTS_DIR / f"rrs_{DEFAULT_METRO}.md")

    print(f"\n✓ Phase 5 complete. μ={float(scores['global_mean'][0]):.3f}, "
          f"mean RRS={scores['rrs'].mean():.3f}, "
          f"mean |Δ|={scores['rrs_delta'].abs().mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
