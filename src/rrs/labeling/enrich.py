"""Build the per-review enriched DataFrame the labeling functions operate on.

One DuckDB query computes per-business burst days, per-user gap statistics, and joins
business/user context to each review. Text-level features (length, exclamation count,
caps ratio) are added in Polars afterwards because regex on 1.9M rows is faster there.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import polars as pl

from rrs.config import DB_PATH
from rrs.labeling.constants import BURST_LOOKBACK_DAYS, BURST_SIGMA

# Note: `friends` arrives as a comma-separated user_id string ("None" when empty), so we
# postpone parsing it to Polars where vectorized string ops are cheap. Same for `text`.
ENRICH_SQL = f"""
WITH biz AS (
    SELECT business_id, stars AS business_mean_stars,
           review_count AS business_total_reviews
    FROM businesses
),
usr AS (
    SELECT user_id, review_count AS user_review_count, yelping_since,
           friends, fans, average_stars AS user_avg_stars, compliment_photos
    FROM users
),
daily_counts AS (
    SELECT business_id, CAST(date AS DATE) AS day, count(*) AS n
    FROM reviews GROUP BY 1, 2
),
burst_days AS (
    SELECT business_id, day FROM (
        SELECT business_id, day, n,
               avg(n) OVER w AS baseline_mean,
               stddev_samp(n) OVER w AS baseline_sd,
               count(*) OVER w AS baseline_n
        FROM daily_counts
        WINDOW w AS (
            PARTITION BY business_id ORDER BY day
            ROWS BETWEEN {BURST_LOOKBACK_DAYS} PRECEDING AND 1 PRECEDING
        )
    )
    WHERE baseline_n >= 3
      AND baseline_sd > 0
      AND n > baseline_mean + {BURST_SIGMA} * baseline_sd
),
gaps AS (
    SELECT user_id,
           extract(epoch FROM
                   (date - lag(date) OVER (PARTITION BY user_id ORDER BY date)))
               / 86400.0 AS gap_days
    FROM reviews
),
user_gap_stats AS (
    SELECT user_id,
           avg(gap_days) AS gap_mean,
           stddev_samp(gap_days) AS gap_std,
           count(gap_days) AS n_gaps
    FROM gaps WHERE gap_days IS NOT NULL
    GROUP BY user_id
),
user_first_week AS (
    -- count of reviews this user posted within 7 days of account creation, within metro.
    -- A coordinated bot-style ramp tends to hit ≥5 here; ordinary users hit 0 or 1.
    SELECT u.user_id, count(r.review_id) AS first_week_count
    FROM users u LEFT JOIN reviews r
      ON r.user_id = u.user_id
     AND r.date >= u.yelping_since
     AND r.date <  u.yelping_since + INTERVAL '7 day'
    GROUP BY u.user_id
)
SELECT
    r.review_id, r.user_id, r.business_id, r.stars, r.text, r.date,
    b.business_mean_stars, b.business_total_reviews,
    u.user_review_count, u.yelping_since,
    u.friends, u.fans, u.user_avg_stars, u.compliment_photos,
    date_diff('day', u.yelping_since, r.date) AS account_age_days_at_review,
    CASE WHEN bd.day IS NOT NULL THEN 1 ELSE 0 END AS in_burst_window,
    gs.gap_std, gs.gap_mean, gs.n_gaps,
    coalesce(fw.first_week_count, 0) AS first_week_count
FROM reviews r
LEFT JOIN biz b              ON r.business_id = b.business_id
LEFT JOIN usr u              ON r.user_id = u.user_id
LEFT JOIN burst_days bd      ON r.business_id = bd.business_id
                             AND CAST(r.date AS DATE) = bd.day
LEFT JOIN user_gap_stats gs  ON r.user_id = gs.user_id
LEFT JOIN user_first_week fw ON r.user_id = fw.user_id
"""


def _add_text_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add char-level review-text features the content LFs need."""
    text = pl.col("text").fill_null("")
    text_len = text.str.len_chars()
    return df.with_columns(
        text_len.alias("text_len"),
        text.str.count_matches("!").alias("exclamation_count"),
        # caps_ratio = (uppercase letters) / (total chars); 0 for empty text
        pl.when(text_len > 0)
        .then(text.str.count_matches(r"[A-Z]") / text_len)
        .otherwise(0.0)
        .alias("caps_ratio"),
    )


def _add_friend_count(df: pl.DataFrame) -> pl.DataFrame:
    """Yelp stores friends as a comma-separated user_id string, 'None' when empty."""
    friends = pl.col("friends").fill_null("None")
    return df.with_columns(
        pl.when((friends == "None") | (friends == ""))
        .then(0)
        .otherwise(friends.str.count_matches(",") + 1)
        .alias("friend_count")
    )


def build_enriched(db_path: Path = DB_PATH) -> pl.DataFrame:
    """Run the enrichment query and add Polars-side features. Returns one row per review."""
    if not db_path.exists():
        raise SystemExit(f"No DuckDB file at {db_path}. Run `python -m rrs.ingest` first.")
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        # Total review-text bytes for Philadelphia (~1.1 GB) overflow Arrow's standard
        # 2 GB string buffer once joined with other text columns. Switch to large buffers.
        con.execute("SET arrow_large_buffer_size = true")
        df = pl.from_arrow(con.execute(ENRICH_SQL).fetch_arrow_table())
    finally:
        con.close()
    df = _add_text_features(df)
    df = _add_friend_count(df)
    return df
