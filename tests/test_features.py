"""Unit tests for Phase 3 feature engineering — synthetic inputs, no DB needed.

The DuckDB-driven aggregations are integration-tested by running the full pipeline; here
we cover the pure-function pieces: polars text features, friend-count parsing, the
within-group max-similarity helper, and entropy/safe-divide edge cases.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from rrs.features.content import _add_polars_text_features
from rrs.features.embeddings import max_sim_within_groups
from rrs.features.reviewer import _parse_friend_count


def test_polars_text_features_basic_counts():
    df = pl.DataFrame({"text": ["Hello world!", "WOW SO GOOD!!!", "i went there with my friends"]})
    out = _add_polars_text_features(df)
    assert out["char_length"].to_list() == [12, 14, 28]
    assert out["word_count"].to_list()  == [2, 3, 6]
    # First-person hits "i" (case-insensitive boundary) and "my" → 2.
    fp = out["first_person_ratio"].to_list()
    assert fp[0] == 0.0
    assert fp[2] == pytest.approx(2 / 6, rel=1e-3)
    # Exclamation/caps ratios are non-negative and bounded by 1.
    for col in ("exclamation_ratio", "caps_ratio"):
        vals = out[col].to_list()
        assert all(0 <= v <= 1 for v in vals), (col, vals)


def test_polars_text_features_empty_text_safe():
    df = pl.DataFrame({"text": ["", None, "x"]})
    out = _add_polars_text_features(df)
    # No NaNs and no divide-by-zeros.
    def is_finite_number(v):
        return isinstance(v, (int, float)) and not (isinstance(v, float) and v != v)

    cols = ("char_length", "word_count", "sentence_count",
            "exclamation_ratio", "caps_ratio", "first_person_ratio")
    for col in cols:
        vals = out[col].to_list()
        assert not any(v is None for v in vals), col
        assert all(is_finite_number(v) for v in vals), col


def test_parse_friend_count_none_empty_and_list():
    df = pl.DataFrame({
        "friends_raw": ["None", "", "a", "a,b,c,d,e", None],
    })
    out = _parse_friend_count(df)
    assert out["friend_count"].to_list() == [0, 0, 1, 5, 0]
    assert "friends_raw" not in out.columns


def test_max_sim_within_groups_self_excluded():
    # Two unit vectors per group. Cosine = 1 only with the "self" copy, which we exclude.
    # Group A: two identical vectors → max sim to other = 1.0 each.
    # Group B: two orthogonal vectors → max sim to other = 0.0.
    e = np.array([
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ], dtype=np.float32)
    keys = np.array(["A", "A", "B", "B"])
    out = max_sim_within_groups(keys, e)
    assert out[0] == pytest.approx(1.0)
    assert out[1] == pytest.approx(1.0)
    assert out[2] == pytest.approx(0.0)
    assert out[3] == pytest.approx(0.0)


def test_max_sim_within_groups_singleton_is_zero():
    e = np.eye(3, dtype=np.float32)
    keys = np.array(["X", "Y", "Z"])  # each appears once
    out = max_sim_within_groups(keys, e)
    assert (out == 0).all()
