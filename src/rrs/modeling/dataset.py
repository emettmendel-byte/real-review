"""Phase 4 dataset assembly — join features to weak labels and split by time.

The suspicion model is trained on the discriminative features from Phase 3
(`features/reviews.parquet`) with the Snorkel `p_suspicious` soft label from Phase 2
(`labels/weak_labels.parquet`) as the regression target. This is the standard
weak-supervision "end model": the LabelModel denoises the noisy LF votes, then a
discriminative model learns to generalize from features to the denoised probability,
including over reviews where every LF abstained.

Two design decisions baked in here:

* **Time split, never random.** Train on reviews dated before 2020, test on 2020+.
  Random splits leak signal (a user's other reviews, a burst's siblings) across the
  boundary. See `SPLIT_DATE`.

* **Leakage-aware feature selection.** The Phase 3 per-user aggregates (`total_reviews`,
  `rating_variance`, `fan_count`, ...) are computed over each user's *full* history
  through the Jan-2022 snapshot, so a 2018 training review's features already reflect the
  user's 2021 behavior. That is a known temporal leak. We keep those aggregates (they
  carry the bulk of the account-quality signal and the plan's scope assumes them) but
  drop the one feature that is *purely* snapshot-relative and has a clean point-in-time
  replacement: `account_age_days_snapshot` is dropped in favour of
  `account_age_days_at_review`. The remaining leak is documented in the Phase 4 report and
  in `LEAKY_AGGREGATE_FEATURES` so downstream readers can reason about it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import polars as pl

from rrs.config import ROOT

FEATURES_PATH = ROOT / "features" / "reviews.parquet"
LABELS_PATH = ROOT / "labels" / "weak_labels.parquet"

# Reviews dated strictly before this go to train; on/after go to test. A real datetime
# (not a polars expr) so it works both in `.filter()` and in Series comparisons.
SPLIT_DATE = datetime(2020, 1, 1)

# Probability at/above which a soft label is treated as a hard "suspicious" positive.
# Used only for ranking metrics (AUC/AP/precision@k) and early-stopping — the model is
# *trained* on the continuous probability, not this threshold.
HARD_LABEL_THRESHOLD = 0.5

# Columns that are never features: identifiers, raw datetimes (the split uses `date`
# but the model must not see the timestamp itself), and the label/target columns.
ID_COLS = ("review_id", "user_id", "business_id")
DATETIME_COLS = ("date", "yelping_since")

# Snapshot-relative feature with a clean point-in-time replacement — dropped to reduce
# the worst of the temporal leak. See module docstring.
DROPPED_LEAKY_COLS = ("account_age_days_snapshot",)

# Per-user aggregates computed over full history through the snapshot. NOT dropped, but
# flagged so the report and any future point-in-time refactor can find them.
LEAKY_AGGREGATE_FEATURES = (
    "total_reviews",
    "fan_count",
    "photo_count",
    "friend_count",
    "reviews_per_month",
    "rating_variance",
    "rating_skew",
    "avg_review_length",
    "fraction_extreme_ratings",
    "posting_hour_entropy",
)


def feature_columns(all_columns: list[str]) -> list[str]:
    """The ordered list of model feature columns: everything except ids, datetimes,
    dropped-leaky columns, and the label columns (which live in the labels frame)."""
    excluded = set(ID_COLS) | set(DATETIME_COLS) | set(DROPPED_LEAKY_COLS)
    return [c for c in all_columns if c not in excluded]


@dataclass
class Dataset:
    """A time-split modeling dataset. `X_*` are feature frames aligned to `*_meta`
    (review_id + date), `soft_*` is the continuous target, `hard_*` its binarization."""

    feature_cols: list[str]
    X_train: pl.DataFrame
    soft_train: pl.Series
    X_test: pl.DataFrame
    soft_test: pl.Series
    train_meta: pl.DataFrame
    test_meta: pl.DataFrame

    @property
    def hard_train(self) -> pl.Series:
        return (self.soft_train >= HARD_LABEL_THRESHOLD).cast(pl.Int8)

    @property
    def hard_test(self) -> pl.Series:
        return (self.soft_test >= HARD_LABEL_THRESHOLD).cast(pl.Int8)


def load_joined(
    features_path: Path = FEATURES_PATH,
    labels_path: Path = LABELS_PATH,
) -> pl.DataFrame:
    """Inner-join features to their weak label on review_id. Every review has both, so
    inner == left here; inner is defensive against a partial rebuild."""
    feats = pl.read_parquet(features_path)
    labels = pl.read_parquet(labels_path).select(["review_id", "p_suspicious"])
    joined = feats.join(labels, on="review_id", how="inner")
    if joined.height != feats.height:
        # Surface a partial/mismatched label set rather than silently training on a subset.
        raise ValueError(
            f"feature/label row mismatch: {feats.height:,} features but "
            f"{joined.height:,} joined to a label. Rebuild labels or features."
        )
    return joined


def time_split(joined: pl.DataFrame, split_date: datetime = SPLIT_DATE) -> Dataset:
    """Partition the joined frame into pre-/post-split feature matrices + targets."""
    cols = feature_columns(joined.columns)
    # `p_suspicious` is the target, not a feature.
    cols = [c for c in cols if c != "p_suspicious"]

    train = joined.filter(pl.col("date") < split_date)
    test = joined.filter(pl.col("date") >= split_date)
    meta_cols = ["review_id", "date"]
    return Dataset(
        feature_cols=cols,
        X_train=train.select(cols),
        soft_train=train.get_column("p_suspicious"),
        X_test=test.select(cols),
        soft_test=test.get_column("p_suspicious"),
        train_meta=train.select(meta_cols),
        test_meta=test.select(meta_cols),
    )
