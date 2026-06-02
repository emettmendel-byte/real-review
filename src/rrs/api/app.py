"""Phase 6 — FastAPI service for the Real Rating Score.

Three v1 endpoints over the Phase 1–5 outputs:

    GET /businesses/search?q=&city=&limit=        — name/city search + RRS
    GET /businesses/{business_id}                 — full RRS bundle for one business
    GET /businesses/{business_id}/reviews         — reviews + p_fake + top_signals

`p_fake` and the RRS are a **second opinion, never a verdict** (see the plan's "Honest
limitations"). The `top_signals` strings describe *signals*, not accusations, and never
name a user.

Run:  PYTHONPATH=src uv run uvicorn rrs.api.app:app --port 8011
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Query

from rrs.api import data
from rrs.api.signals import top_signals as build_top_signals
from rrs.config import DEFAULT_METRO


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load booster / explainer / parquet caches once before serving any request.
    data.warm()
    yield


app = FastAPI(
    title="Real Rating Score API",
    version="1.0",
    summary="Adjusted, fake-review-aware ratings for businesses on the Yelp Open Dataset.",
    description=(
        "RRS and per-review `p_fake` are a **second opinion, not a verdict**. Flagged "
        "reviews are surfaced with plain-language signals, never accusations against a user."
    ),
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check + which metro's data is loaded."""
    return {"status": "ok", "metro": DEFAULT_METRO}


@app.get("/businesses/search")
def search_businesses(
    q: str = Query(..., min_length=1, description="Case-insensitive substring of the name."),
    city: str | None = Query(None, description="Optional exact (case-insensitive) city filter."),
    limit: int = Query(20, ge=1, le=100),
) -> list[dict]:
    """Search businesses by name (and optional city), ordered by Yelp review count desc.

    Each hit carries the business's Yelp rating/review-count and its RRS (the adjusted,
    fake-review-aware rating) for an at-a-glance comparison."""
    sql = """
        SELECT business_id, name, city, state, stars, review_count
        FROM businesses
        WHERE lower(name) LIKE '%' || lower(?) || '%'
    """
    params: list[object] = [q]
    if city:
        sql += " AND lower(city) = lower(?)"
        params.append(city)
    sql += " ORDER BY review_count DESC LIMIT ?"
    params.append(limit)

    con = data.connect()
    try:
        rows = con.execute(sql, params).fetchall()
    finally:
        con.close()

    rrs_by_id = data.rrs_map([r[0] for r in rows])
    return [
        {
            "business_id": r[0],
            "name": r[1],
            "city": r[2],
            "state": r[3],
            "yelp_rating": r[4],
            "yelp_review_count": r[5],
            "rrs": rrs_by_id.get(r[0]),
        }
        for r in rows
    ]


@app.get("/businesses/{business_id}")
def get_business(business_id: str) -> dict:
    """Full RRS bundle for one business: Yelp rating vs RRS, the confidence interval, and
    the transparency counts (flagged / authentic). 404 if the id is unknown."""
    con = data.connect()
    try:
        row = con.execute(
            """
            SELECT business_id, name, address, city, state, categories, stars, review_count
            FROM businesses WHERE business_id = ?
            """,
            [business_id],
        ).fetchone()
    finally:
        con.close()
    if row is None:
        raise HTTPException(status_code=404, detail="Unknown business_id")

    rrs = data.rrs_for_business(business_id)
    return {
        "business_id": row[0],
        "name": row[1],
        "address": row[2],
        "city": row[3],
        "state": row[4],
        "categories": row[5],
        "yelp_rating": row[6],
        "yelp_review_count": row[7],
        # RRS bundle — None if the business has no scored reviews (defensive; all do).
        "rrs": _maybe(rrs, "rrs"),
        "rrs_ci_low": _maybe(rrs, "rrs_ci_low"),
        "rrs_ci_high": _maybe(rrs, "rrs_ci_high"),
        "pct_flagged": _maybe(rrs, "pct_flagged"),
        "n_flagged": _maybe(rrs, "n_flagged"),
        "n_authentic_reviews": _maybe(rrs, "n_authentic_reviews"),
        "n_reviews": _maybe(rrs, "n_reviews"),
    }


@app.get("/businesses/{business_id}/reviews")
def get_reviews(
    business_id: str,
    include_flags: bool = Query(True, description="Compute per-review top_signals (SHAP)."),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> list[dict]:
    """Reviews for a business with each review's `p_fake` and, when `include_flags=true`,
    the top 3 plain-language `top_signals` explaining the score. 404 if the id is unknown.

    Signals describe *why a review looks suspicious*, never who wrote it — `p_fake` is a
    second opinion, not a verdict."""
    con = data.connect()
    try:
        exists = con.execute(
            "SELECT 1 FROM businesses WHERE business_id = ? LIMIT 1", [business_id]
        ).fetchone()
        if exists is None:
            raise HTTPException(status_code=404, detail="Unknown business_id")
        rows = con.execute(
            """
            SELECT review_id, stars, text, date
            FROM reviews WHERE business_id = ?
            ORDER BY date DESC
            LIMIT ? OFFSET ?
            """,
            [business_id, limit, offset],
        ).fetchall()
    finally:
        con.close()

    review_ids = [r[0] for r in rows]
    p_fake = data.p_fake_map(review_ids)
    signals = _signals_for(review_ids) if include_flags else {}

    return [
        {
            "review_id": r[0],
            "stars": r[1],
            "text": r[2],
            "date": r[3].isoformat() if r[3] is not None else None,
            "p_fake": p_fake.get(r[0]),
            "top_signals": signals.get(r[0], []),
        }
        for r in rows
    ]


# ---- helpers ---------------------------------------------------------------------------

def _maybe(d: dict | None, key: str):
    return None if d is None else d[key]


def _signals_for(review_ids: list[str]) -> dict[str, list[str]]:
    """review_id → top-3 incriminating signal strings, via the cached SHAP TreeExplainer.

    One explainer call over the whole batch of feature rows (fast for a few hundred rows)."""
    feats = data.feature_rows(review_ids)
    if feats.is_empty():
        return {}

    names = data.feature_names()
    ids = feats["review_id"].to_list()
    # Align feature matrix to the booster's feature order.
    matrix = feats.select(names).to_numpy().astype(np.float64)

    sv = data.explainer().shap_values(matrix)
    # Binary cross-entropy booster → TreeExplainer returns a single (n, n_features) array.
    if isinstance(sv, list):
        sv = sv[-1]
    sv = np.asarray(sv)

    out: dict[str, list[str]] = {}
    for i, rid in enumerate(ids):
        out[rid] = build_top_signals(names, matrix[i].tolist(), sv[i].tolist())
    return out
