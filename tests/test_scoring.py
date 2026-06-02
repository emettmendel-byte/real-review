"""Unit tests for Phase 5 RRS aggregation — pure functions, synthetic inputs."""

from __future__ import annotations

import polars as pl
import pytest

from rrs.scoring import (
    PRIOR_WEIGHT,
    global_trust_weighted_mean,
    score_businesses,
    wilson_interval,
)


def test_wilson_interval_brackets_estimate_and_clamps():
    lo, hi = wilson_interval(0.5, 100)
    assert 0.0 <= lo < 0.5 < hi <= 1.0
    # Smaller n → wider interval.
    lo_small, hi_small = wilson_interval(0.5, 10)
    assert (hi_small - lo_small) > (hi - lo)
    # Extremes stay in [0, 1].
    lo0, hi0 = wilson_interval(0.0, 5)
    assert lo0 == 0.0 and 0.0 < hi0 <= 1.0
    lo1, hi1 = wilson_interval(1.0, 5)
    assert hi1 == 1.0 and 0.0 <= lo1 < 1.0


def test_wilson_interval_zero_n_is_point():
    assert wilson_interval(0.4, 0) == (0.4, 0.4)


def test_global_trust_weighted_mean_downweights_fakes():
    # Two 5★ reviews (one almost certainly fake) and one genuine 1★.
    df = pl.DataFrame({"stars": [5.0, 5.0, 1.0], "p_fake": [0.0, 0.99, 0.0]})
    mu = global_trust_weighted_mean(df)
    naive = df["stars"].mean()  # 3.667
    # The fake 5★ barely counts, so the trusted mean is pulled toward the 1★ and 5★ that
    # are real → (5 + 1) / 2 = 3.0, well below the naive 3.667.
    assert mu == pytest.approx(3.0, abs=0.05)
    assert mu < naive


def test_score_businesses_shrinks_thin_evidence_toward_global():
    # Business B has a single trusted 5★; with K=10 prior it should sit far below 5,
    # close to the global mean. Business A has many trusted reviews and barely shrinks.
    rows = []
    for _ in range(50):
        rows.append(("A", 5.0, 0.0))
    rows.append(("B", 5.0, 0.0))
    # Add a spread so the global mean isn't 5.
    for _ in range(20):
        rows.append(("C", 2.0, 0.0))
    df = pl.DataFrame(rows, schema=["business_id", "stars", "p_fake"], orient="row")
    scores = score_businesses(df).sort("business_id")
    by_id = {r["business_id"]: r for r in scores.iter_rows(named=True)}
    mu = by_id["A"]["global_mean"]

    # A: 50 trusted 5★ → close to 5, only mildly shrunk.
    assert by_id["A"]["rrs"] > 4.5
    # B: one 5★ → strongly pulled toward the global mean.
    assert by_id["B"]["rrs"] < by_id["A"]["rrs"]
    assert abs(by_id["B"]["rrs"] - mu) < abs(5.0 - mu)
    # n_eff equals summed trust; all p_fake=0 here so it equals the review count.
    assert by_id["A"]["n_eff"] == pytest.approx(50.0)
    assert by_id["A"]["n_reviews"] == 50


def test_score_businesses_flag_counts_and_ci_order():
    df = pl.DataFrame(
        {
            "business_id": ["X", "X", "X", "X"],
            "stars": [5.0, 5.0, 1.0, 1.0],
            "p_fake": [0.9, 0.1, 0.2, 0.8],  # two above the 0.5 flag threshold
        }
    )
    row = score_businesses(df).row(0, named=True)
    assert row["n_reviews"] == 4
    assert row["n_flagged"] == 2
    assert row["n_authentic_reviews"] == 2
    assert row["pct_flagged"] == pytest.approx(0.5)
    assert row["rrs_ci_low"] <= row["rrs"] <= row["rrs_ci_high"]
    assert 1.0 <= row["rrs_ci_low"] and row["rrs_ci_high"] <= 5.0
    # Fully-flagged-or-not, the prior keeps it finite and in range.
    assert 1.0 <= row["rrs"] <= 5.0
    assert PRIOR_WEIGHT == 10.0
