"""Phase 6 API tests.

Two layers:

* **Pure-function unit tests** for the SHAP → text mapping (`signals`), with synthetic
  inputs and no IO — these always run.

* **`TestClient` smoke tests** that hit each endpoint against the real local data
  (`data/yelp.duckdb` + the `models/`/`features/` parquet files). They are skipped if
  those artifacts (or the `ml`/`api` extras) are missing, so the suite still passes on a
  bare checkout. They never touch the multi-GB raw JSON.
"""

from __future__ import annotations

import importlib.util

import pytest

from rrs.api import signals
from rrs.api.data import (
    BOOSTER_PATH,
    EXPLAINER_PATH,
    FEATURES_PATH,
    PREDICTIONS_PATH,
    RRS_SCORES_PATH,
)
from rrs.config import DB_PATH

# ---- pure unit tests: SHAP → signal strings --------------------------------------------


def test_render_signal_uses_actual_value():
    assert signals.render_signal("account_age_days_at_review", 4.0) == (
        "Account created 4 days before the review"
    )
    assert signals.render_signal("max_sim_to_user_history", 0.94) == (
        "Text 94% similar to another review by the same user"
    )
    assert signals.render_signal("max_sim_to_business_reviews", 0.88) == (
        "Text 88% similar to another review of this business"
    )
    assert signals.render_signal("is_in_burst_window", 1.0) == (
        "Posted during a burst of reviews on this business"
    )
    assert signals.render_signal("total_reviews", 1.0) == (
        "This is among the reviewer's very first reviews"
    )
    assert signals.render_signal("total_reviews", 3.0) == "Reviewer has only 3 total reviews"
    assert signals.render_signal("friend_count", 0.0) == (
        "Reviewer has no friends on the platform"
    )


def test_render_signal_unknown_feature_has_safe_fallback():
    out = signals.render_signal("some_new_feature", 3.0)
    assert "some_new_feature" not in out  # never leak the raw column name verbatim
    assert out == "Atypical value for some new feature"


def test_top_signals_keeps_only_positive_and_ranks_by_contribution():
    names = ["account_age_days_at_review", "friend_count", "total_reviews", "stars"]
    values = [4.0, 0.0, 3.0, 5.0]
    shap = [2.0, -0.5, 1.5, 0.3]  # friend_count negative → excluded
    out = signals.top_signals(names, values, shap, k=3)
    assert out == [
        "Account created 4 days before the review",  # highest +contribution
        "Reviewer has only 3 total reviews",
        "An extreme 5-star rating",
    ]


def test_top_signals_returns_fewer_than_k_when_few_positive():
    names = ["total_reviews", "friend_count"]
    values = [3.0, 0.0]
    shap = [1.2, -3.0]
    assert signals.top_signals(names, values, shap, k=3) == ["Reviewer has only 3 total reviews"]


def test_top_signals_all_strings():
    names = ["total_reviews", "caps_ratio", "exclamation_ratio"]
    values = [1.0, 0.5, 0.4]
    shap = [1.0, 0.5, 0.2]
    out = signals.top_signals(names, values, shap)
    assert all(isinstance(s, str) and s for s in out)


# ---- integration smoke tests against real local data -----------------------------------

_ARTIFACTS = [DB_PATH, RRS_SCORES_PATH, PREDICTIONS_PATH, FEATURES_PATH, BOOSTER_PATH,
              EXPLAINER_PATH]
_HAS_DATA = all(p.exists() for p in _ARTIFACTS)
_HAS_DEPS = all(importlib.util.find_spec(m) is not None for m in ("fastapi", "shap", "lightgbm"))

requires_local = pytest.mark.skipif(
    not (_HAS_DATA and _HAS_DEPS),
    reason="needs built data/yelp.duckdb + models/features parquet + ml/api extras",
)


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from rrs.api.app import app

    with TestClient(app) as c:  # triggers lifespan → cache warm
        yield c


@pytest.fixture(scope="module")
def sample_business_id():
    import polars as pl

    # Pick a business with enough reviews that top_signals is meaningful.
    df = pl.read_parquet(RRS_SCORES_PATH, columns=["business_id", "n_reviews"])
    return df.sort("n_reviews", descending=True).row(0, named=True)["business_id"]


def test_booster_features_match_renderers():
    """Every model feature must have a renderer (so no raw column name can leak)."""
    if not (_HAS_DEPS and BOOSTER_PATH.exists()):
        pytest.skip("needs booster + lightgbm")
    import lightgbm as lgb

    feats = lgb.Booster(model_file=str(BOOSTER_PATH)).feature_name()
    missing = [f for f in feats if f not in signals.RENDERERS]
    assert not missing, f"features without a renderer: {missing}"


@requires_local
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


@requires_local
def test_search_shape(client):
    r = client.get("/businesses/search", params={"q": "pizza", "limit": 5})
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, list) and len(body) <= 5
    if body:
        b = body[0]
        assert set(b) == {
            "business_id", "name", "city", "state",
            "yelp_rating", "yelp_review_count", "rrs",
        }
        assert "pizza" in b["name"].lower()


@requires_local
def test_get_business_shape_and_ci(client, sample_business_id):
    r = client.get(f"/businesses/{sample_business_id}")
    assert r.status_code == 200
    b = r.json()
    assert set(b) == {
        "business_id", "name", "address", "city", "state", "categories",
        "yelp_rating", "yelp_review_count", "rrs", "rrs_ci_low", "rrs_ci_high",
        "pct_flagged", "n_flagged", "n_authentic_reviews", "n_reviews",
    }
    assert b["business_id"] == sample_business_id
    assert b["rrs_ci_low"] <= b["rrs"] <= b["rrs_ci_high"]


@requires_local
def test_get_business_404(client):
    assert client.get("/businesses/__nope__").status_code == 404


@requires_local
def test_reviews_with_flags(client, sample_business_id):
    r = client.get(
        f"/businesses/{sample_business_id}/reviews",
        params={"include_flags": "true", "limit": 25},
    )
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, list) and body
    rv = body[0]
    assert set(rv) == {"review_id", "stars", "text", "date", "p_fake", "top_signals"}
    assert isinstance(rv["top_signals"], list)
    assert all(isinstance(s, str) for s in rv["top_signals"])
    assert len(rv["top_signals"]) <= 3
    # p_fake is a probability second opinion.
    assert rv["p_fake"] is None or 0.0 <= rv["p_fake"] <= 1.0


@requires_local
def test_reviews_without_flags_skips_signals(client, sample_business_id):
    r = client.get(
        f"/businesses/{sample_business_id}/reviews",
        params={"include_flags": "false", "limit": 5},
    )
    assert r.status_code == 200
    assert all(rv["top_signals"] == [] for rv in r.json())


@requires_local
def test_reviews_404(client):
    assert client.get("/businesses/__nope__/reviews").status_code == 404
