"""Phase 3 — feature engineering.

Three families joined per-review and written to `features/reviews.parquet`:

- `reviewer.py`  per-user aggregates: tenure, rating moments, social, posting hour entropy.
- `content.py`   per-review text features: lengths, ratios, VADER sentiment, Flesch.
- `context.py`   per-review context: stars deltas, prev-review lag, cumulative count, burst flag.
- `embeddings.py` MiniLM (`all-MiniLM-L6-v2`) embedding + per-user and per-business
  max-cosine-to-other-review-by-same-key scalars. Embeddings persist to
  `features/embeddings.npy` (memmap) + `embeddings_index.parquet` so Phase 4 can
  decide whether to consume them raw, PCA-reduced, or only via the scalars.

    uv run python -m rrs.features.build
"""
