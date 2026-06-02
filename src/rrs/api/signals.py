"""SHAP → human-readable "signal" strings (Phase 6).

The trust-building differentiator of the API: for a review the model scored as
possibly inauthentic, surface *why* in plain language. We never name or accuse a
user — we describe the signal, not a verdict (see the plan's "Honest limitations").

The mechanism:

* The Phase 4 `shap.TreeExplainer` gives a per-feature contribution for each review,
  in **log-odds space**. A *positive* contribution pushes `p_fake` UP — i.e. it is an
  *incriminating* signal. We keep only positive contributions and take the top 3.
* Each surviving feature is rendered from its **actual value** for that review via a
  phrasing function in `RENDERERS`. Every one of the 30 model features has a renderer
  (`feature_name() == list(RENDERERS)` is asserted by the tests), so there is always a
  sensible string, never a raw column name.

This module is deliberately free of any ML / IO imports so the mapping is unit-testable
with synthetic inputs.
"""

from __future__ import annotations

from collections.abc import Callable

# How many signals we surface per review.
TOP_K = 3


def _plural(n: float, singular: str, plural: str) -> str:
    return singular if abs(n - 1.0) < 1e-9 else plural


def _days(value: float) -> str:
    d = int(round(value))
    if d <= 0:
        return "Account created the same day as the review"
    return f"Account created {d} {_plural(d, 'day', 'days')} before the review"


def _sim_user(value: float) -> str:
    return f"Text {round(value * 100)}% similar to another review by the same user"


def _sim_business(value: float) -> str:
    return f"Text {round(value * 100)}% similar to another review of this business"


def _burst(value: float) -> str:
    return "Posted during a burst of reviews on this business"


def _total_reviews(value: float) -> str:
    n = int(round(value))
    if n <= 1:
        return "This is among the reviewer's very first reviews"
    return f"Reviewer has only {n} total reviews"


def _friend_count(value: float) -> str:
    n = int(round(value))
    if n == 0:
        return "Reviewer has no friends on the platform"
    return f"Reviewer has few friends on the platform ({n})"


def _fan_count(value: float) -> str:
    n = int(round(value))
    if n == 0:
        return "Reviewer has no fans on the platform"
    return f"Reviewer has few fans on the platform ({n})"


def _photo_count(value: float) -> str:
    n = int(round(value))
    if n == 0:
        return "Reviewer has never posted a photo"
    return f"Reviewer has posted few photos ({n})"


def _reviews_per_month(value: float) -> str:
    return f"Reviewer posts an unusually high volume of reviews ({value:.1f}/month)"


def _stars(value: float) -> str:
    s = int(round(value))
    return f"An extreme {s}-star rating"


def _stars_delta_business(value: float) -> str:
    direction = "above" if value > 0 else "below"
    return f"Rating is {abs(value):.1f} stars {direction} this business's average"


def _stars_delta_user(value: float) -> str:
    direction = "above" if value > 0 else "below"
    return f"Rating is {abs(value):.1f} stars {direction} the reviewer's own average"


def _hours_since_prev(value: float) -> str:
    if value < 1.0:
        return "Posted minutes after the reviewer's previous review here"
    h = int(round(value))
    return f"Posted only {h} {_plural(h, 'hour', 'hours')} after a prior review here"


def _business_review_count(value: float) -> str:
    n = int(round(value))
    return f"Among the first {n} reviews this business received"


def _rating_variance(value: float) -> str:
    return "Reviewer's ratings are unusually uniform across businesses"


def _rating_skew(value: float) -> str:
    return "Reviewer's rating history is skewed toward the extremes"


def _avg_review_length(value: float) -> str:
    return f"Reviewer's reviews are unusually short on average ({int(round(value))} chars)"


def _fraction_extreme(value: float) -> str:
    return f"{round(value * 100)}% of the reviewer's ratings are 1 or 5 stars"


def _posting_hour_entropy(value: float) -> str:
    return "Reviewer posts at suspiciously regular times of day"


def _account_age_snapshot(value: float) -> str:
    # Not in the model feature set, but kept for completeness/robustness.
    return _days(value)


def _char_length(value: float) -> str:
    n = int(round(value))
    return f"Unusually short review text ({n} {_plural(n, 'character', 'characters')})"


def _word_count(value: float) -> str:
    n = int(round(value))
    return f"Unusually few words ({n} {_plural(n, 'word', 'words')})"


def _sentence_count(value: float) -> str:
    n = int(round(value))
    return f"Only {n} {_plural(n, 'sentence', 'sentences')} of text"


def _exclamation_ratio(value: float) -> str:
    return "Heavy use of exclamation marks"


def _caps_ratio(value: float) -> str:
    return "Heavy use of capital letters"


def _first_person_ratio(value: float) -> str:
    return "Unusual use of first-person language"


def _vader_compound(value: float) -> str:
    tone = "strongly positive" if value > 0 else "strongly negative"
    return f"Text sentiment is {tone}"


def _vader_pos(value: float) -> str:
    return "Text is overwhelmingly positive in tone"


def _vader_neg(value: float) -> str:
    return "Text is overwhelmingly negative in tone"


def _vader_neu(value: float) -> str:
    return "Text is unusually flat / neutral in tone"


def _flesch(value: float) -> str:
    return "Text readability is atypical for genuine reviews"


# feature_name -> (value -> human-readable string). Order is the booster's feature order
# so the test that asserts coverage can compare against `booster.feature_name()`.
RENDERERS: dict[str, Callable[[float], str]] = {
    "stars": _stars,
    "char_length": _char_length,
    "word_count": _word_count,
    "sentence_count": _sentence_count,
    "exclamation_ratio": _exclamation_ratio,
    "caps_ratio": _caps_ratio,
    "first_person_ratio": _first_person_ratio,
    "vader_compound": _vader_compound,
    "vader_pos": _vader_pos,
    "vader_neg": _vader_neg,
    "vader_neu": _vader_neu,
    "flesch_reading_ease": _flesch,
    "stars_delta_from_business_mean": _stars_delta_business,
    "stars_delta_from_user_mean": _stars_delta_user,
    "hours_since_prev_review_on_business": _hours_since_prev,
    "business_review_count_at_time": _business_review_count,
    "is_in_burst_window": _burst,
    "total_reviews": _total_reviews,
    "fan_count": _fan_count,
    "photo_count": _photo_count,
    "account_age_days_snapshot": _account_age_snapshot,
    "reviews_per_month": _reviews_per_month,
    "rating_variance": _rating_variance,
    "rating_skew": _rating_skew,
    "avg_review_length": _avg_review_length,
    "fraction_extreme_ratings": _fraction_extreme,
    "posting_hour_entropy": _posting_hour_entropy,
    "friend_count": _friend_count,
    "account_age_days_at_review": _days,
    "max_sim_to_user_history": _sim_user,
    "max_sim_to_business_reviews": _sim_business,
}


def render_signal(feature: str, value: float) -> str:
    """Render one (feature, value) pair into a plain-language signal string.

    Falls back to a generic phrasing for any feature without an explicit renderer so the
    caller never leaks a raw column name into the response."""
    fn = RENDERERS.get(feature)
    if fn is None:
        return f"Atypical value for {feature.replace('_', ' ')}"
    return fn(float(value))


def top_signals(
    feature_names: list[str],
    values: list[float],
    shap_values: list[float],
    k: int = TOP_K,
) -> list[str]:
    """Top-`k` incriminating signals for one review.

    `feature_names`, `values`, and `shap_values` are aligned per-feature lists (SHAP in
    log-odds space, positive == pushes `p_fake` up). We keep only features with a strictly
    positive SHAP contribution, sort by that contribution descending, and render the top
    `k`. Fewer than `k` positive contributions → fewer strings (possibly empty)."""
    ranked = sorted(
        (
            (sv, name, val)
            for name, val, sv in zip(feature_names, values, shap_values, strict=True)
            if sv > 0
        ),
        key=lambda t: t[0],
        reverse=True,
    )
    return [render_signal(name, val) for _, name, val in ranked[:k]]
