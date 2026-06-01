"""Vectorized labeling functions.

Each LF takes the enriched per-review Polars DataFrame and returns an Int8 Series of
ABSTAIN/AUTHENTIC/SUSPICIOUS — same length as the input. Writing them as polars
expressions instead of per-row callbacks lets us label 1.9M reviews in seconds rather
than the ~10 minutes a `PandasLFApplier` loop would cost.

Each LF deliberately has both a SUSPICIOUS branch and an AUTHENTIC branch (returning
ABSTAIN only when the signal genuinely can't tell). Snorkel's LabelModel needs LFs to
vote on both classes — one-sided LFs degrade to majority vote.
"""

from __future__ import annotations

import polars as pl

from rrs.labeling.constants import (
    ABSTAIN,
    ACCOUNT_FIRST_WEEK_THRESHOLD,
    AUTHENTIC,
    BREVITY_AUTHENTIC_CHARS,
    BREVITY_CHARS,
    DUP_SIM_AUTHENTIC,
    DUP_SIM_SUSPICIOUS,
    NEW_ACCOUNT_AUTHENTIC_DAYS,
    NEW_ACCOUNT_DAYS,
    NO_SOCIAL_AUTHENTIC_FANS,
    NO_SOCIAL_AUTHENTIC_FRIENDS,
    ONE_SHOT_AUTHENTIC_FLOOR,
    ONE_SHOT_THRESHOLD,
    RATING_DEV_THRESHOLD,
    REGULARITY_AUTHENTIC_CV,
    REGULARITY_CV,
    REGULARITY_MIN_GAPS,
    SUSPICIOUS,
    TEMPLATE_AUTHENTIC_LEN,
    TEMPLATE_CAPS_RATIO,
    TEMPLATE_EXCLAM,
    TEMPLATE_MAX_LEN,
)


def _three_way(suspicious: pl.Expr, authentic: pl.Expr) -> pl.Expr:
    """Build a SUSPICIOUS / AUTHENTIC / ABSTAIN expression from two bool predicates.

    SUSPICIOUS takes precedence over AUTHENTIC when both happen to be true.
    """
    return (
        pl.when(suspicious)
        .then(SUSPICIOUS)
        .when(authentic)
        .then(AUTHENTIC)
        .otherwise(ABSTAIN)
        .cast(pl.Int8)
    )


# --- Behavioral ----------------------------------------------------------------------

def lf_burst(df: pl.DataFrame) -> pl.Series:
    """Review posted during a >3σ daily-volume spike for that business."""
    susp = pl.col("in_burst_window") == 1
    auth = (pl.col("in_burst_window") == 0) & (pl.col("business_total_reviews") >= 20)
    return df.select(_three_way(susp, auth).alias("lf_burst"))["lf_burst"]


def lf_one_shot_extreme(df: pl.DataFrame) -> pl.Series:
    """User has very few lifetime reviews and this one is 1★ or 5★."""
    extreme = pl.col("stars").is_in([1.0, 5.0])
    susp = (pl.col("user_review_count") <= ONE_SHOT_THRESHOLD) & extreme
    auth = pl.col("user_review_count") >= ONE_SHOT_AUTHENTIC_FLOOR
    return df.select(_three_way(susp, auth).alias("lf_one_shot_extreme"))["lf_one_shot_extreme"]


def lf_rating_deviation(df: pl.DataFrame) -> pl.Series:
    """Stars deviate from BOTH the business mean AND the reviewer's own history."""
    biz_dev = (pl.col("stars") - pl.col("business_mean_stars")).abs()
    user_dev = (pl.col("stars") - pl.col("user_avg_stars")).abs()
    susp = (biz_dev > RATING_DEV_THRESHOLD) & (user_dev > RATING_DEV_THRESHOLD)
    auth = (biz_dev < 0.5) & (user_dev < 0.5)
    return df.select(_three_way(susp, auth).alias("lf_rating_deviation"))["lf_rating_deviation"]


def lf_temporal_regularity(df: pl.DataFrame) -> pl.Series:
    """User posts at suspiciously regular intervals (low CV of inter-review gaps)."""
    enough = pl.col("n_gaps") >= REGULARITY_MIN_GAPS
    cv = pl.col("gap_std") / pl.col("gap_mean")
    susp = enough & cv.is_not_null() & (cv < REGULARITY_CV)
    auth = enough & cv.is_not_null() & (cv > REGULARITY_AUTHENTIC_CV)
    name = "lf_temporal_regularity"
    return df.select(_three_way(susp, auth).alias(name))[name]


# --- Content -------------------------------------------------------------------------

def lf_duplicate_text(df: pl.DataFrame) -> pl.Series:
    """Near-duplicate phrasing of another review by the same user.

    Reads the precomputed `max_dup_sim` column produced by `dup_text.compute_max_sim`.
    """
    susp = pl.col("max_dup_sim") > DUP_SIM_SUSPICIOUS
    auth = (pl.col("max_dup_sim") < DUP_SIM_AUTHENTIC) & (pl.col("user_review_count") >= 5)
    return df.select(_three_way(susp, auth).alias("lf_duplicate_text"))["lf_duplicate_text"]


def lf_template_text(df: pl.DataFrame) -> pl.Series:
    """Short, shouty review (lots of caps or exclamations) on an extreme rating."""
    shouty = (pl.col("caps_ratio") > TEMPLATE_CAPS_RATIO) | (
        pl.col("exclamation_count") >= TEMPLATE_EXCLAM
    )
    susp = (
        (pl.col("text_len") < TEMPLATE_MAX_LEN)
        & shouty
        & pl.col("stars").is_in([1.0, 5.0])
    )
    auth = pl.col("text_len") > TEMPLATE_AUTHENTIC_LEN
    return df.select(_three_way(susp, auth).alias("lf_template_text"))["lf_template_text"]


def lf_extreme_brevity(df: pl.DataFrame) -> pl.Series:
    """Almost no text on a 5★ rating."""
    susp = (pl.col("text_len") < BREVITY_CHARS) & (pl.col("stars") == 5.0)
    auth = (pl.col("text_len") > BREVITY_AUTHENTIC_CHARS) & (pl.col("stars") == 5.0)
    return df.select(_three_way(susp, auth).alias("lf_extreme_brevity"))["lf_extreme_brevity"]


# --- Account quality -----------------------------------------------------------------

def lf_new_account(df: pl.DataFrame) -> pl.Series:
    """Account was less than a month old when the review was posted."""
    age = pl.col("account_age_days_at_review")
    susp = age.is_not_null() & (age < NEW_ACCOUNT_DAYS)
    auth = age.is_not_null() & (age > NEW_ACCOUNT_AUTHENTIC_DAYS)
    return df.select(_three_way(susp, auth).alias("lf_new_account"))["lf_new_account"]


def lf_no_social(df: pl.DataFrame) -> pl.Series:
    """Reviewer has zero social-graph presence."""
    susp = (
        (pl.col("friend_count") == 0)
        & (pl.col("fans") == 0)
        & (pl.col("compliment_photos") == 0)
    )
    auth = (pl.col("fans") >= NO_SOCIAL_AUTHENTIC_FANS) | (
        pl.col("friend_count") >= NO_SOCIAL_AUTHENTIC_FRIENDS
    )
    return df.select(_three_way(susp, auth).alias("lf_no_social"))["lf_no_social"]


def lf_account_burst(df: pl.DataFrame) -> pl.Series:
    """Account posted many reviews within its first week — bot-style ramp pattern."""
    susp = pl.col("first_week_count") > ACCOUNT_FIRST_WEEK_THRESHOLD
    auth = pl.col("first_week_count") <= 1
    return df.select(_three_way(susp, auth).alias("lf_account_burst"))["lf_account_burst"]


# Ordered list — index in the label matrix matches this order. Keep it stable for
# the LabelModel to interpret correctly across runs.
LFS: list = [
    lf_burst,
    lf_one_shot_extreme,
    lf_rating_deviation,
    lf_temporal_regularity,
    lf_duplicate_text,
    lf_template_text,
    lf_extreme_brevity,
    lf_new_account,
    lf_no_social,
    lf_account_burst,
]

LF_NAMES: list[str] = [fn.__name__ for fn in LFS]
