"""Per-user near-duplicate detection for `lf_duplicate_text`.

For every user with ≥2 reviews in the metro, fit a small word-level TF-IDF model on just
that user's review texts and compute pairwise cosine similarity. Returns each review's
max similarity to any *other* review by the same user. The cheap per-user TF-IDF is
~100× faster than embedding 1.9M reviews with a sentence transformer, and at threshold
0.9 it catches the near-verbatim duplicates the LF cares about. Embedding-based
similarity is deferred to Phase 3 features, where it does richer work.

One-shot reviewers get `max_dup_sim = 0` by definition (nothing to compare against).
"""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


def _per_user_max_sim(texts: list[str]) -> list[float]:
    """Max cosine similarity to any other text in the group, per row. 0 for singletons."""
    n = len(texts)
    if n < 2:
        return [0.0] * n
    cleaned = [t if t else " " for t in texts]
    try:
        vec = TfidfVectorizer(min_df=1, lowercase=True, ngram_range=(1, 2))
        X = vec.fit_transform(cleaned)
    except ValueError:
        # All-empty / stopword-only texts produce an empty vocabulary.
        return [0.0] * n
    sim = linear_kernel(X)  # (n, n) cosine since TF-IDF rows are L2-normalized
    np.fill_diagonal(sim, -1.0)
    return sim.max(axis=1).clip(min=0.0).tolist()


def compute_max_sim(df: pl.DataFrame) -> pl.Series:
    """Return an f32 column aligned to `df` rows with the per-user max duplicate-sim."""
    # Sort by user_id so we can groupby in a single pass and reassemble in input order.
    df_with_idx = df.with_row_index("_row_idx").select(
        ["_row_idx", "user_id", "text"]
    )

    out = np.zeros(df_with_idx.height, dtype=np.float32)
    # `partition_by` returns one DataFrame per user.
    for grp in df_with_idx.partition_by("user_id", maintain_order=False):
        idx = grp["_row_idx"].to_numpy()
        sims = _per_user_max_sim(grp["text"].to_list())
        out[idx] = sims
    return pl.Series("max_dup_sim", out, dtype=pl.Float32)
