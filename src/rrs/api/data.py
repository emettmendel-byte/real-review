"""Cached data-access layer for the Phase 6 API.

Everything expensive is loaded **once** (module-level, memoised with `lru_cache`) and
shared across requests: the LightGBM booster, the pickled SHAP `TreeExplainer`, the
per-review `p_fake` predictions, and the per-business RRS table. DuckDB is opened
**read-only** — the API never writes — via short-lived per-call connections, which is
cheap and side-steps the single-writer/threading constraints of a long-lived handle.

The two parquet lookups (`predictions`, `rrs_scores`) are held as Polars frames; row
lookups go through small helpers here so the route handlers stay thin.
"""

from __future__ import annotations

import pickle
from functools import lru_cache
from typing import Any

import duckdb
import lightgbm as lgb
import polars as pl

from rrs.config import DB_PATH, ROOT

PREDICTIONS_PATH = ROOT / "models" / "predictions.parquet"
RRS_SCORES_PATH = ROOT / "models" / "rrs_scores.parquet"
FEATURES_PATH = ROOT / "features" / "reviews.parquet"
BOOSTER_PATH = ROOT / "models" / "lgbm_suspicion.txt"
EXPLAINER_PATH = ROOT / "models" / "shap_explainer.pkl"


@lru_cache(maxsize=1)
def booster() -> lgb.Booster:
    return lgb.Booster(model_file=str(BOOSTER_PATH))


@lru_cache(maxsize=1)
def feature_names() -> list[str]:
    """Authoritative model feature order — what the explainer and parquet must align to."""
    return list(booster().feature_name())


@lru_cache(maxsize=1)
def explainer() -> Any:
    with open(EXPLAINER_PATH, "rb") as fh:
        return pickle.load(fh)


@lru_cache(maxsize=1)
def predictions() -> pl.DataFrame:
    """review_id → p_fake for every review."""
    return pl.read_parquet(PREDICTIONS_PATH, columns=["review_id", "p_fake"])


@lru_cache(maxsize=1)
def rrs_scores() -> pl.DataFrame:
    """Per-business RRS bundle (Phase 5 output)."""
    return pl.read_parquet(RRS_SCORES_PATH)


def warm() -> None:
    """Eagerly populate every cache (called once at startup so the first request is fast)."""
    booster()
    feature_names()
    explainer()
    predictions()
    rrs_scores()


def connect() -> duckdb.DuckDBPyConnection:
    """A short-lived read-only DuckDB connection. Caller closes it."""
    return duckdb.connect(str(DB_PATH), read_only=True)


# ---- per-business / per-review lookups -------------------------------------------------

def rrs_for_business(business_id: str) -> dict[str, Any] | None:
    row = rrs_scores().filter(pl.col("business_id") == business_id)
    if row.is_empty():
        return None
    return row.row(0, named=True)


def rrs_map(business_ids: list[str]) -> dict[str, float]:
    """business_id → rrs for a set of ids (used by search to attach RRS in one pass)."""
    if not business_ids:
        return {}
    sub = rrs_scores().filter(pl.col("business_id").is_in(business_ids))
    return dict(zip(sub["business_id"], sub["rrs"], strict=True))


def p_fake_map(review_ids: list[str]) -> dict[str, float]:
    if not review_ids:
        return {}
    sub = predictions().filter(pl.col("review_id").is_in(review_ids))
    return {rid: float(p) for rid, p in zip(sub["review_id"], sub["p_fake"], strict=True)}


def feature_rows(review_ids: list[str]) -> pl.DataFrame:
    """Model-feature rows for the given reviews, columns aligned to `feature_names()`,
    plus `review_id` as the first column. Order is *not* guaranteed to match the input."""
    if not review_ids:
        return pl.DataFrame()
    cols = ["review_id", *feature_names()]
    feats = pl.read_parquet(FEATURES_PATH, columns=cols)
    return feats.filter(pl.col("review_id").is_in(review_ids))
