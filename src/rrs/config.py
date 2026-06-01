"""Central configuration: filesystem paths and metro definitions.

The Yelp Open Dataset (Jan 2022 release) covers a fixed set of metros. Notably it
does NOT contain Los Angeles — the "LA" state code in the data is Louisiana
(New Orleans). The only sizeable California metro is Santa Barbara. We default to
Philadelphia, the largest metro, for the strongest fake-review signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Repo root is two levels above this file: src/rrs/config.py -> <root>
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "Yelp JSON" / "yelp_dataset"
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
DB_PATH = DATA_DIR / "yelp.duckdb"

RAW_FILES: dict[str, Path] = {
    "business": RAW_DIR / "yelp_academic_dataset_business.json",
    "review": RAW_DIR / "yelp_academic_dataset_review.json",
    "user": RAW_DIR / "yelp_academic_dataset_user.json",
    "tip": RAW_DIR / "yelp_academic_dataset_tip.json",
    "checkin": RAW_DIR / "yelp_academic_dataset_checkin.json",
}


@dataclass(frozen=True)
class Metro:
    """A target metro. Businesses are filtered by state, optionally narrowed by city."""

    key: str
    name: str
    states: tuple[str, ...]
    cities: tuple[str, ...] | None = None  # if set, AND city IN (...)


# In this dataset each metro maps cleanly to a small set of states. The Philadelphia
# metro spans SE Pennsylvania plus its NJ and DE suburbs (Cherry Hill, Wilmington);
# all PA/NJ/DE rows in the dump belong to that metro.
METROS: dict[str, Metro] = {
    "philadelphia": Metro("philadelphia", "Philadelphia", ("PA", "NJ", "DE")),
    "santa_barbara": Metro(
        "santa_barbara",
        "Santa Barbara",
        ("CA",),
        ("Santa Barbara", "Goleta", "Carpinteria", "Montecito", "Isla Vista", "Summerland"),
    ),
    "new_orleans": Metro("new_orleans", "New Orleans", ("LA",)),
    "tampa": Metro("tampa", "Tampa", ("FL",)),
    "tucson": Metro("tucson", "Tucson", ("AZ",)),
    "indianapolis": Metro("indianapolis", "Indianapolis", ("IN",)),
    "nashville": Metro("nashville", "Nashville", ("TN",)),
    "reno": Metro("reno", "Reno", ("NV",)),
    "saint_louis": Metro("saint_louis", "Saint Louis", ("MO",)),
    "boise": Metro("boise", "Boise", ("ID",)),
}

DEFAULT_METRO = "philadelphia"


def get_metro(key: str) -> Metro:
    try:
        return METROS[key]
    except KeyError:
        valid = ", ".join(sorted(METROS))
        raise SystemExit(f"Unknown metro '{key}'. Choose one of: {valid}") from None
