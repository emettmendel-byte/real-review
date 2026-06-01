"""Run all labeling functions, combine via Snorkel's LabelModel, and persist.

    uv run python -m rrs.labeling.apply
    uv run python -m rrs.labeling.apply --db data/yelp.duckdb --out labels/

Outputs:
    labels/weak_labels.parquet                — review_id, p_suspicious, per-LF votes
    reports/labels_<metro>.md                 — LFAnalysis + top-100 audit sample
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl

from rrs.config import DB_PATH, REPORTS_DIR, ROOT
from rrs.labeling.constants import ABSTAIN, SUSPICIOUS
from rrs.labeling.dup_text import compute_max_sim
from rrs.labeling.enrich import build_enriched
from rrs.labeling.lfs import LF_NAMES, LFS

DEFAULT_OUT = ROOT / "labels"


def _time(label: str):
    """Tiny context manager for step timing — visible progress on a multi-minute run."""

    class _T:
        def __enter__(self_):
            self_.t0 = time.perf_counter()
            print(f"  → {label} ...", flush=True)
            return self_

        def __exit__(self_, *a):
            print(f"    {time.perf_counter() - self_.t0:.1f}s", flush=True)

    return _T()


def apply_lfs(df: pl.DataFrame) -> np.ndarray:
    """Stack each vectorized LF into a (n_reviews, n_lfs) Int8 matrix."""
    cols = [fn(df).to_numpy().astype(np.int8) for fn in LFS]
    return np.column_stack(cols)


def _metro_label(db_path: Path) -> str:
    import duckdb

    try:
        con = duckdb.connect(str(db_path), read_only=True)
        try:
            row = con.execute("SELECT name FROM meta LIMIT 1").fetchone()
            return (row[0] if row else "unknown").lower().replace(" ", "_")
        finally:
            con.close()
    except Exception:
        return "unknown"


def fit_label_model(L: np.ndarray, lf_names: list[str], seed: int = 42) -> np.ndarray:
    """Fit Snorkel LabelModel; return p_suspicious per row.

    Falls back to MajorityLabelVoter if LabelModel can't fit (e.g. all-abstain rows or
    a degenerate matrix). The fallback is also a useful sanity baseline.
    """
    from snorkel.labeling import LFAnalysis
    from snorkel.labeling.model import LabelModel, MajorityLabelVoter

    summary = LFAnalysis(L=L).lf_summary()
    summary.index = lf_names  # name the rows so the audit table is readable
    print("\n  Per-LF summary (Snorkel LFAnalysis):")
    print(summary.to_string())

    try:
        lm = LabelModel(cardinality=2, verbose=False)
        lm.fit(L_train=L, n_epochs=500, log_freq=200, seed=seed)
        proba = lm.predict_proba(L)  # shape (N, 2), columns are [AUTHENTIC, SUSPICIOUS]
        return proba[:, SUSPICIOUS].astype(np.float32)
    except Exception as e:
        print(f"  LabelModel failed ({e!s}); falling back to MajorityLabelVoter.")
        mv = MajorityLabelVoter(cardinality=2)
        proba = mv.predict_proba(L)
        return proba[:, SUSPICIOUS].astype(np.float32)


def write_audit(
    df: pl.DataFrame,
    p_suspicious: np.ndarray,
    L: np.ndarray,
    lf_names: list[str],
    out_md: Path,
    top_n: int = 100,
) -> None:
    """Markdown report: per-LF stats + the `top_n` highest-p_suspicious reviews."""
    out_md.parent.mkdir(parents=True, exist_ok=True)
    n = len(df)
    pct_flagged = float((p_suspicious > 0.5).mean()) * 100
    lf_cov = (L != ABSTAIN).mean(axis=0)
    lf_sus = (L == SUSPICIOUS).mean(axis=0)

    lines = [
        "# Weak labels — audit",
        "",
        f"- Reviews scored: **{n:,}**",
        f"- Flagged (p_suspicious > 0.5): **{pct_flagged:.2f}%**",
        f"- p_suspicious quantiles: "
        f"P50={np.quantile(p_suspicious, 0.5):.3f} · "
        f"P90={np.quantile(p_suspicious, 0.9):.3f} · "
        f"P99={np.quantile(p_suspicious, 0.99):.3f}",
        "",
        "## Per-LF coverage and SUSPICIOUS-vote rate",
        "",
        "| LF | coverage | suspicious-rate |",
        "|---|---:|---:|",
    ]
    for name, c, s in zip(lf_names, lf_cov, lf_sus, strict=True):
        lines.append(f"| `{name}` | {c * 100:.2f}% | {s * 100:.2f}% |")
    lines.append("")

    lines.append(f"## Top {top_n} highest-confidence suspicious reviews")
    lines.append("")
    order = np.argsort(-p_suspicious)[:top_n]
    audit = df.select(
        ["review_id", "user_id", "business_id", "stars", "date", "text"]
    ).to_numpy()
    for rank, i in enumerate(order, 1):
        rid, uid, bid, stars, date, text = audit[i]
        votes = ", ".join(
            f"`{lf_names[j]}`" for j in range(L.shape[1]) if L[i, j] == SUSPICIOUS
        ) or "(none — model inferred from LF correlations)"
        snippet = (text or "")[:200].replace("\n", " ")
        lines += [
            f"### {rank}. p_suspicious={p_suspicious[i]:.3f} · {stars}★ · {date}",
            f"- review `{rid}` by user `{uid}` on business `{bid}`",
            f"- LFs voting SUSPICIOUS: {votes}",
            f"- text: {snippet}{'…' if text and len(text) > 200 else ''}",
            "",
        ]
    out_md.write_text("\n".join(lines))
    print(f"\nWrote audit → {out_md}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 — weak supervision labels.")
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-n", type=int, default=100, help="audit sample size")
    args = parser.parse_args()

    metro = _metro_label(args.db)
    print(f"Generating weak labels for metro '{metro}' from {args.db}\n")

    with _time("enrich (DuckDB + Polars text features)"):
        df = build_enriched(args.db)
        print(f"    n_reviews = {df.height:,}")

    with _time("per-user duplicate-text similarity"):
        df = df.with_columns(compute_max_sim(df))

    with _time(f"apply {len(LFS)} labeling functions (vectorized)"):
        L = apply_lfs(df)
        all_abstain = int((L == ABSTAIN).all(axis=1).sum())
        print(f"    label matrix shape {L.shape}; rows where every LF abstained: {all_abstain:,}")

    with _time("fit Snorkel LabelModel"):
        p_suspicious = fit_label_model(L, LF_NAMES, seed=args.seed)

    with _time("write parquet"):
        votes_df = pl.DataFrame({name: L[:, i] for i, name in enumerate(LF_NAMES)})
        out_df = (
            df.select(["review_id"])
            .with_columns(
                pl.Series("p_suspicious", p_suspicious),
                pl.Series("predicted_label", (p_suspicious > 0.5).astype(np.int8)),
            )
            .hstack(votes_df)
        )
        args.out.mkdir(parents=True, exist_ok=True)
        parquet_path = args.out / "weak_labels.parquet"
        out_df.write_parquet(parquet_path)
        print(f"    wrote {parquet_path} ({out_df.height:,} rows, {len(out_df.columns)} cols)")

    write_audit(
        df,
        p_suspicious,
        L,
        LF_NAMES,
        out_md=REPORTS_DIR / f"labels_{metro}.md",
        top_n=args.top_n,
    )


if __name__ == "__main__":
    main()
