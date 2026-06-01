"""Label values and tunable thresholds for Phase 2 labeling functions.

Snorkel convention: each LF returns one of these integers per row, and the LabelModel
combines votes across LFs into a probabilistic label.
"""

from __future__ import annotations

# Snorkel label values. ABSTAIN means "I have no opinion on this review."
ABSTAIN = -1
AUTHENTIC = 0
SUSPICIOUS = 1

# --- Behavioral ---
BURST_LOOKBACK_DAYS = 14   # rolling window for the per-business spike baseline
BURST_SIGMA = 3.0          # how many σ above baseline counts as a burst
ONE_SHOT_THRESHOLD = 2     # ≤N lifetime reviews + extreme stars → suspicious
ONE_SHOT_AUTHENTIC_FLOOR = 10
RATING_DEV_THRESHOLD = 2.0
REGULARITY_CV = 0.1        # coefficient of variation of inter-review gaps below this = suspicious
REGULARITY_AUTHENTIC_CV = 0.5
REGULARITY_MIN_GAPS = 3    # need ≥4 reviews (3 gaps) for the variance to be meaningful

# --- Content ---
DUP_SIM_SUSPICIOUS = 0.9
DUP_SIM_AUTHENTIC = 0.3
TEMPLATE_MAX_LEN = 200
TEMPLATE_CAPS_RATIO = 0.2
TEMPLATE_EXCLAM = 3
TEMPLATE_AUTHENTIC_LEN = 500
BREVITY_CHARS = 20
BREVITY_AUTHENTIC_CHARS = 200

# --- Account quality ---
NEW_ACCOUNT_DAYS = 30
NEW_ACCOUNT_AUTHENTIC_DAYS = 365
ACCOUNT_FIRST_WEEK_THRESHOLD = 5
NO_SOCIAL_FANS = 0
NO_SOCIAL_AUTHENTIC_FANS = 5
NO_SOCIAL_AUTHENTIC_FRIENDS = 20
