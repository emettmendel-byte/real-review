"""Unit tests for Phase 4 dataset assembly — pure functions, no DB or model fit.

The LightGBM/Optuna/SHAP training path is integration-tested by the full run; here we
cover the deterministic pieces: feature-column selection (id/datetime/leaky exclusions),
the time split, hard-label binarization, and the soft cross-entropy loss.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import polars as pl

from rrs.modeling.dataset import (
    DROPPED_LEAKY_COLS,
    HARD_LABEL_THRESHOLD,
    feature_columns,
    time_split,
)
from rrs.modeling.train import _soft_logloss


def _toy_joined() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "review_id": ["a", "b", "c", "d"],
            "user_id": ["u1", "u1", "u2", "u3"],
            "business_id": ["x", "y", "x", "z"],
            "date": [
                datetime(2018, 6, 1),
                datetime(2019, 6, 1),
                datetime(2020, 6, 1),
                datetime(2021, 6, 1),
            ],
            "yelping_since": [datetime(2017, 1, 1)] * 4,
            "stars": [5.0, 1.0, 4.0, 2.0],
            "char_length": [10, 20, 30, 40],
            "account_age_days_snapshot": [100, 200, 300, 400],
            "account_age_days_at_review": [10, 20, 30, 40],
            "p_suspicious": [0.9, 0.1, 0.6, 0.2],
        }
    )


def test_feature_columns_excludes_ids_datetimes_and_leaky():
    cols = ["review_id", "user_id", "business_id", "date", "yelping_since",
            "stars", "char_length", "account_age_days_snapshot",
            "account_age_days_at_review", "p_suspicious"]
    feats = feature_columns(cols)
    for dropped in ("review_id", "user_id", "business_id", "date", "yelping_since",
                    *DROPPED_LEAKY_COLS):
        assert dropped not in feats, dropped
    # Kept: the real features (and p_suspicious, which time_split removes separately).
    assert "stars" in feats
    assert "char_length" in feats
    assert "account_age_days_at_review" in feats


def test_time_split_partitions_at_2020_and_excludes_target():
    ds = time_split(_toy_joined())
    # 2018, 2019 → train; 2020, 2021 → test.
    assert ds.X_train.height == 2
    assert ds.X_test.height == 2
    # The target is never a feature.
    assert "p_suspicious" not in ds.feature_cols
    # ids/datetimes/leaky excluded.
    for bad in ("review_id", "date", "account_age_days_snapshot"):
        assert bad not in ds.feature_cols
    # Soft targets line up with the partition.
    assert ds.soft_train.to_list() == [0.9, 0.1]
    assert ds.soft_test.to_list() == [0.6, 0.2]
    # Meta carries review_id + date for the audit/ablation joins.
    assert ds.train_meta.columns == ["review_id", "date"]


def test_hard_label_thresholds_soft():
    ds = time_split(_toy_joined())
    # threshold 0.5: train [0.9, 0.1] → [1, 0]; test [0.6, 0.2] → [1, 0].
    assert ds.hard_train.to_list() == [1, 0]
    assert ds.hard_test.to_list() == [1, 0]
    assert HARD_LABEL_THRESHOLD == 0.5


def test_soft_logloss_is_minimized_at_truth():
    soft = np.array([0.9, 0.1, 0.5, 0.0, 1.0])
    perfect = _soft_logloss(soft, soft.copy())
    worse = _soft_logloss(soft, np.full_like(soft, 0.5))
    assert perfect < worse
    # Symmetric, finite, non-negative.
    assert perfect >= 0.0
    assert np.isfinite(_soft_logloss(soft, np.array([1.0, 0.0, 0.0, 1.0, 0.0])))
