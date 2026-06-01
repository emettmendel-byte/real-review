"""Per-user reviewer features.

`compute_reviewer_features(con)` returns one row per user_id with the 11 features the
plan calls out — to be left-joined onto the review-level feature table. All aggregates
are computed in DuckDB and the `friends` field is parsed in Polars (it's a comma-
separated user_id string, large for power users).

A note on time-leakage: the user-level aggregates (rating variance, fraction extreme, …)
are computed over the user's *whole* in-metro history, which can include reviews
written after the review being scored. The Yelp dump is a 2022 snapshot, so even
`users.review_count` (lifetime) carries the same leak. This is consistent with the
plan's framing and acceptable for a research-grade weak-supervision pipeline; if we
ever serve live, these need to be recomputed at request time.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import polars as pl

from rrs.config import DB_PATH

# Snapshot date — last review in the dump is 2022-01-19. Using this rather than `now()`
# avoids drifting account-age math each run.
SNAPSHOT_DATE = "DATE '2022-01-19'"

REVIEWER_SQL = f"""
WITH user_review_aggs AS (
    SELECT
        user_id,
        var_samp(stars)                                       AS rating_variance,
        skewness(stars)                                       AS rating_skew,
        avg(length(text))                                     AS avg_review_length,
        avg(CASE WHEN stars IN (1.0, 5.0) THEN 1.0 ELSE 0.0 END)
                                                              AS fraction_extreme_ratings
    FROM reviews
    GROUP BY user_id
),
user_hour_dist AS (
    SELECT user_id, extract(hour FROM date) AS h, count(*)::DOUBLE AS n
    FROM reviews
    GROUP BY user_id, h
),
user_entropy AS (
    -- Shannon entropy of the 24-bin hour-of-day posting distribution. Coordinated bots
    -- often post in a narrow window; ordinary users spread out.
    SELECT user_id,
           -sum( (n/total) * ln(n/total) ) AS posting_hour_entropy
    FROM (
        SELECT user_id, n, sum(n) OVER (PARTITION BY user_id) AS total
        FROM user_hour_dist
    )
    GROUP BY user_id
)
SELECT
    u.user_id,
    u.review_count                                            AS total_reviews,
    u.fans                                                    AS fan_count,
    u.compliment_photos                                       AS photo_count,
    u.friends                                                 AS friends_raw,
    date_diff('day', u.yelping_since, {SNAPSHOT_DATE})        AS account_age_days_snapshot,
    u.review_count / greatest(
        date_diff('day', u.yelping_since, {SNAPSHOT_DATE}) / 30.44, 1.0
    )                                                         AS reviews_per_month,
    coalesce(a.rating_variance,         0.0)                  AS rating_variance,
    coalesce(a.rating_skew,             0.0)                  AS rating_skew,
    coalesce(a.avg_review_length,       0.0)                  AS avg_review_length,
    coalesce(a.fraction_extreme_ratings, 0.0)                 AS fraction_extreme_ratings,
    coalesce(e.posting_hour_entropy,    0.0)                  AS posting_hour_entropy,
    u.yelping_since
FROM users u
LEFT JOIN user_review_aggs a ON u.user_id = a.user_id
LEFT JOIN user_entropy     e ON u.user_id = e.user_id
"""


def _parse_friend_count(df: pl.DataFrame) -> pl.DataFrame:
    friends = pl.col("friends_raw").fill_null("None")
    return df.with_columns(
        pl.when((friends == "None") | (friends == ""))
        .then(0)
        .otherwise(friends.str.count_matches(",") + 1)
        .alias("friend_count")
    ).drop("friends_raw")


def compute_reviewer_features(db_path: Path = DB_PATH) -> pl.DataFrame:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        con.execute("SET arrow_large_buffer_size = true")
        df = pl.from_arrow(con.execute(REVIEWER_SQL).fetch_arrow_table())
    finally:
        con.close()
    return _parse_friend_count(df)
