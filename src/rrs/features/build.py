"""Phase 3 orchestrator — joins the three feature families and writes the parquet.

    uv run python -m rrs.features.build                         # full Philadelphia run
    uv run python -m rrs.features.build --skip-embeddings       # everything except MiniLM
    uv run python -m rrs.features.build --sample 50000          # debug on a sample

Outputs:
    features/reviews.parquet            — one row per review with all numeric features
    features/embeddings.npy             — (N, 384) float32 array (memmap-friendly)
    features/embeddings_index.parquet   — review_id → row mapping for embeddings.npy

The reviews parquet is a single file sorted by business_id. The plan's wording was
"partitioned by business", but with 44.8K businesses that's 44.8K tiny parquet dirs —
not actually useful. Sorting + parquet row-group stats give the same predicate-pushdown
benefit without the filesystem mess. Phase 4 can `pl.scan_parquet(...).filter(...)`.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import duckdb
import polars as pl

from rrs.config import DB_PATH, ROOT
from rrs.features.content import compute_text_features
from rrs.features.context import compute_context_features
from rrs.features.embeddings import (
    compute_similarities,
    encode_texts,
    estimate_runtime_hours,
    write_embedding_index,
)
from rrs.features.reviewer import compute_reviewer_features

DEFAULT_OUT = ROOT / "features"


def _time(label: str):
    """Step timing — visible progress on a multi-stage run."""

    class _T:
        def __enter__(self_):
            self_.t0 = time.perf_counter()
            print(f"\n→ {label}", flush=True)
            return self_

        def __exit__(self_, *a):
            print(f"  ({time.perf_counter() - self_.t0:.1f}s)", flush=True)

    return _T()


def _load_reviews_skeleton(db_path: Path, sample: int | None = None) -> pl.DataFrame:
    """Just review_id, user_id, business_id, stars, date, text — feeds every other step."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        con.execute("SET arrow_large_buffer_size = true")
        sql = "SELECT review_id, user_id, business_id, stars, date, text FROM reviews"
        if sample:
            # Deterministic sample using DuckDB's hash — keeps tests reproducible.
            sql += f" ORDER BY hash(review_id) LIMIT {int(sample)}"
        return pl.from_arrow(con.execute(sql).fetch_arrow_table())
    finally:
        con.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 — feature engineering.")
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--sample", type=int, default=None, help="debug: subsample N reviews")
    parser.add_argument("--skip-embeddings", action="store_true")
    parser.add_argument("--batch-size", type=int, default=128, help="MiniLM batch size")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    with _time("load review skeleton from DuckDB"):
        reviews = _load_reviews_skeleton(args.db, sample=args.sample)
        n = reviews.height
        print(f"  {n:,} reviews to feature")
        if not args.skip_embeddings:
            eta_h = estimate_runtime_hours(n)
            print(f"  embedding ETA on MPS ≈ {eta_h * 60:.0f} min (rough)")

    with _time("reviewer features (per-user)"):
        rev_feats = compute_reviewer_features(args.db)
        print(f"  {rev_feats.height:,} users × {len(rev_feats.columns)} cols")

    with _time("context features (per-review)"):
        ctx_feats = compute_context_features(args.db)
        if args.sample:
            ctx_feats = ctx_feats.join(reviews.select("review_id"), on="review_id", how="inner")

    with _time("content features — polars text + VADER + Flesch"):
        content_feats = compute_text_features(reviews.select(["review_id", "text"]))
        # We don't keep raw text in the feature table — text lives in DuckDB.
        content_feats = content_feats.drop("text")

    # Join everything we have so far. Embeddings happen separately so we don't
    # double-pay the join when --skip-embeddings is on.
    with _time("join (reviewer + content + context)"):
        features = (
            reviews.select(["review_id", "user_id", "business_id", "stars", "date"])
            .join(content_feats, on="review_id", how="left")
            .join(ctx_feats,     on="review_id", how="left")
            .join(rev_feats,     on="user_id",   how="left")
        )
        # Per-review account age — more meaningful for time-split training than the
        # snapshot-relative version since it doesn't leak future tenure.
        features = features.with_columns(
            ((pl.col("date") - pl.col("yelping_since")).dt.total_days())
            .cast(pl.Int32)
            .alias("account_age_days_at_review")
        )

    if args.skip_embeddings:
        features = features.with_columns(
            pl.lit(None, dtype=pl.Float32).alias("max_sim_to_user_history"),
            pl.lit(None, dtype=pl.Float32).alias("max_sim_to_business_reviews"),
        )
        print("\n(skipped embeddings as requested)")
    else:
        with _time("MiniLM embeddings — sentence-transformers on MPS"):
            embs = encode_texts(
                reviews["text"].to_list(),
                batch_size=args.batch_size,
                out_path=args.out / "embeddings.npy",
            )
            write_embedding_index(reviews["review_id"].to_list(), args.out)

        with _time("max-similarity scalars (per-user, per-business)"):
            max_user, max_biz = compute_similarities(
                embs,
                reviews["user_id"].to_list(),
                reviews["business_id"].to_list(),
            )
            features = features.with_columns(
                pl.Series("max_sim_to_user_history",     max_user, dtype=pl.Float32),
                pl.Series("max_sim_to_business_reviews", max_biz,  dtype=pl.Float32),
            )

    with _time("write features/reviews.parquet (sorted by business_id)"):
        features = features.sort("business_id")
        out_path = args.out / "reviews.parquet"
        features.write_parquet(out_path, statistics=True)
        print(f"  wrote {out_path} — {features.height:,} rows × {len(features.columns)} cols")
        print(f"  size: {out_path.stat().st_size / 1e6:.1f} MB")

    # Quick health summary so the user can eyeball the result.
    print("\nFeature columns:")
    for c in features.columns:
        print(f"  {c:38s}  {features.schema[c]}")


if __name__ == "__main__":
    main()
