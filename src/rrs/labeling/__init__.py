"""Phase 2 — weak supervision.

10 vectorized labeling functions (`lfs.py`) feed Snorkel's LabelModel (`apply.py`)
to produce `labels/weak_labels.parquet` with `p_suspicious` per review.

    uv run python -m rrs.labeling.apply

Each LF takes the enriched per-review DataFrame and returns an Int8 series with values
ABSTAIN=-1, AUTHENTIC=0, SUSPICIOUS=1. LFs are written as polars expressions, not
per-row Python — that takes label generation on 1.93M reviews from minutes to seconds.
"""

from .constants import ABSTAIN, AUTHENTIC, SUSPICIOUS

__all__ = ["ABSTAIN", "AUTHENTIC", "SUSPICIOUS"]
