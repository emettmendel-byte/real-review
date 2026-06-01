"""Unit tests for Phase 2 labeling functions on synthetic Polars rows.

Each LF gets a row that should fire SUSPICIOUS, one that should fire AUTHENTIC, and one
that should ABSTAIN. Keeps the tests independent of any built DuckDB.
"""

from __future__ import annotations

import polars as pl

from rrs.labeling.constants import ABSTAIN, AUTHENTIC, SUSPICIOUS
from rrs.labeling.dup_text import compute_max_sim
from rrs.labeling.lfs import (
    LF_NAMES,
    LFS,
    lf_account_burst,
    lf_burst,
    lf_duplicate_text,
    lf_extreme_brevity,
    lf_new_account,
    lf_no_social,
    lf_one_shot_extreme,
    lf_rating_deviation,
    lf_template_text,
    lf_temporal_regularity,
)


def _row(**kwargs) -> dict:
    """Return one fully populated enrichment row with sensible defaults."""
    base = {
        "review_id": "R",
        "user_id": "U",
        "business_id": "B",
        "stars": 4.0,
        "text": "ok food, would come again",
        "date": None,
        "business_mean_stars": 4.0,
        "business_total_reviews": 100,
        "user_review_count": 20,
        "yelping_since": None,
        "friends": "a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z",
        "fans": 10,
        "user_avg_stars": 4.0,
        "compliment_photos": 2,
        "account_age_days_at_review": 1000,
        "in_burst_window": 0,
        "gap_std": 30.0,
        "gap_mean": 60.0,
        "n_gaps": 10,
        "first_week_count": 1,
        "text_len": 25,
        "exclamation_count": 0,
        "caps_ratio": 0.05,
        "friend_count": 26,
        "max_dup_sim": 0.1,
    }
    base.update(kwargs)
    return base


def _df(*rows: dict) -> pl.DataFrame:
    return pl.DataFrame(list(rows))


def test_lf_burst():
    df = _df(
        _row(in_burst_window=1),
        _row(in_burst_window=0, business_total_reviews=50),
        _row(in_burst_window=0, business_total_reviews=2),
    )
    out = lf_burst(df).to_list()
    assert out == [SUSPICIOUS, AUTHENTIC, ABSTAIN]


def test_lf_one_shot_extreme():
    df = _df(
        _row(user_review_count=1, stars=5.0),
        _row(user_review_count=2, stars=1.0),
        _row(user_review_count=50, stars=5.0),
        _row(user_review_count=5, stars=5.0),
    )
    assert lf_one_shot_extreme(df).to_list() == [SUSPICIOUS, SUSPICIOUS, AUTHENTIC, ABSTAIN]


def test_lf_rating_deviation():
    # biz_mean=4.0, user_avg=4.0; 1★ deviates by 3 from both → SUSPICIOUS
    df = _df(
        _row(stars=1.0),
        _row(stars=4.0),
        _row(stars=3.0),
    )
    assert lf_rating_deviation(df).to_list() == [SUSPICIOUS, AUTHENTIC, ABSTAIN]


def test_lf_temporal_regularity():
    # cv = std/mean.  0.5/60≈0.008 → suspicious;  60/60=1.0 → authentic;  too few gaps → abstain
    df = _df(
        _row(gap_std=0.5, gap_mean=60.0, n_gaps=10),
        _row(gap_std=60.0, gap_mean=60.0, n_gaps=10),
        _row(gap_std=1.0, gap_mean=60.0, n_gaps=1),
    )
    assert lf_temporal_regularity(df).to_list() == [SUSPICIOUS, AUTHENTIC, ABSTAIN]


def test_lf_duplicate_text():
    df = _df(
        _row(max_dup_sim=0.95, user_review_count=10),
        _row(max_dup_sim=0.05, user_review_count=10),
        _row(max_dup_sim=0.5, user_review_count=10),
    )
    assert lf_duplicate_text(df).to_list() == [SUSPICIOUS, AUTHENTIC, ABSTAIN]


def test_lf_template_text():
    df = _df(
        _row(stars=5.0, text_len=40, exclamation_count=5, caps_ratio=0.1),  # shouty short 5★
        _row(stars=4.0, text_len=900, exclamation_count=0, caps_ratio=0.05),  # long → authentic
        _row(stars=5.0, text_len=300, exclamation_count=0, caps_ratio=0.05),  # mid-length, calm
    )
    assert lf_template_text(df).to_list() == [SUSPICIOUS, AUTHENTIC, ABSTAIN]


def test_lf_extreme_brevity():
    df = _df(
        _row(stars=5.0, text_len=8),
        _row(stars=5.0, text_len=400),
        _row(stars=5.0, text_len=80),  # mid-length 5★ → abstain
        _row(stars=3.0, text_len=8),   # short but not 5★ → abstain
    )
    assert lf_extreme_brevity(df).to_list() == [SUSPICIOUS, AUTHENTIC, ABSTAIN, ABSTAIN]


def test_lf_new_account():
    df = _df(
        _row(account_age_days_at_review=10),
        _row(account_age_days_at_review=1000),
        _row(account_age_days_at_review=120),
    )
    assert lf_new_account(df).to_list() == [SUSPICIOUS, AUTHENTIC, ABSTAIN]


def test_lf_no_social():
    df = _df(
        _row(friend_count=0, fans=0, compliment_photos=0, friends="None"),
        _row(friend_count=50, fans=10, compliment_photos=5),
        _row(friend_count=3, fans=1, compliment_photos=0),
    )
    assert lf_no_social(df).to_list() == [SUSPICIOUS, AUTHENTIC, ABSTAIN]


def test_lf_account_burst():
    df = _df(
        _row(first_week_count=10),
        _row(first_week_count=0),
        _row(first_week_count=3),
    )
    assert lf_account_burst(df).to_list() == [SUSPICIOUS, AUTHENTIC, ABSTAIN]


def test_all_lfs_registered():
    """Sanity: LFS list is the 10 functions promised in the plan."""
    assert len(LFS) == 10
    assert set(LF_NAMES) == {
        "lf_burst", "lf_one_shot_extreme", "lf_rating_deviation",
        "lf_temporal_regularity", "lf_duplicate_text", "lf_template_text",
        "lf_extreme_brevity", "lf_new_account", "lf_no_social", "lf_account_burst",
    }


def test_compute_max_sim_within_user():
    """Two near-identical texts by the same user → high similarity; lone user → 0."""
    df = pl.DataFrame({
        "user_id": ["U1", "U1", "U2"],
        "text": [
            "Great pizza, best in town, would come back",
            "Great pizza, best in town, would come back!",
            "Quiet place, mediocre food, slow service",
        ],
    })
    sims = compute_max_sim(df).to_list()
    assert sims[0] > 0.8 and sims[1] > 0.8, sims
    assert sims[2] == 0.0  # only one review for U2
