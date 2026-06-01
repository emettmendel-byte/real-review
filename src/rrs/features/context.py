"""Per-review context features.

`compute_context_features(con)` returns one row per review_id with:
- `stars_delta_from_business_mean`  (signed)
- `stars_delta_from_user_mean`      (signed)
- `hours_since_prev_review_on_business`  (null for the first review on a business)
- `business_review_count_at_time`   running count up to and including this review
- `is_in_burst_window`              1 if this review's day is a >3σ daily-volume spike

The burst-day computation is duplicated from `rrs.labeling.enrich` — keeping the two
phases independent so changes to LF thresholds don't quietly perturb model features.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import polars as pl

from rrs.config import DB_PATH
from rrs.labeling.constants import BURST_LOOKBACK_DAYS, BURST_SIGMA

CONTEXT_SQL = f"""
WITH biz AS (
    SELECT business_id, stars AS biz_mean FROM businesses
),
usr AS (
    SELECT user_id, average_stars AS user_mean FROM users
),
daily_counts AS (
    SELECT business_id, CAST(date AS DATE) AS day, count(*) AS n
    FROM reviews GROUP BY 1, 2
),
burst_days AS (
    SELECT business_id, day FROM (
        SELECT business_id, day, n,
               avg(n)         OVER w AS baseline_mean,
               stddev_samp(n) OVER w AS baseline_sd,
               count(*)       OVER w AS baseline_n
        FROM daily_counts
        WINDOW w AS (
            PARTITION BY business_id ORDER BY day
            ROWS BETWEEN {BURST_LOOKBACK_DAYS} PRECEDING AND 1 PRECEDING
        )
    )
    WHERE baseline_n >= 3 AND baseline_sd > 0
      AND n > baseline_mean + {BURST_SIGMA} * baseline_sd
)
SELECT
    r.review_id,
    r.stars - b.biz_mean                                          AS stars_delta_from_business_mean,
    r.stars - u.user_mean                                         AS stars_delta_from_user_mean,
    extract(epoch FROM (
        r.date - lag(r.date) OVER (PARTITION BY r.business_id ORDER BY r.date)
    )) / 3600.0   AS hours_since_prev_review_on_business,
    row_number() OVER (PARTITION BY r.business_id ORDER BY r.date)
                  AS business_review_count_at_time,
    CASE WHEN bd.day IS NOT NULL THEN 1 ELSE 0 END  AS is_in_burst_window
FROM reviews r
LEFT JOIN biz b         ON r.business_id = b.business_id
LEFT JOIN usr u         ON r.user_id = u.user_id
LEFT JOIN burst_days bd ON r.business_id = bd.business_id
                        AND CAST(r.date AS DATE) = bd.day
"""


def compute_context_features(db_path: Path = DB_PATH) -> pl.DataFrame:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        con.execute("SET arrow_large_buffer_size = true")
        return pl.from_arrow(con.execute(CONTEXT_SQL).fetch_arrow_table())
    finally:
        con.close()
