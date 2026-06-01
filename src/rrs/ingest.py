"""Phase 1 — Data foundation: stream the Yelp JSON dump into DuckDB, one metro at a time.

DuckDB reads the newline-delimited JSON files directly (no separate ETL step). We load
the businesses for the target metro first, then filter the giant review/user files down
to just the reviews on those businesses and the users who wrote them. The 5 GB review and
3 GB user files are streamed and semi-joined, so peak memory stays modest.

Usage:
    uv run python -m rrs.ingest                      # default metro (Philadelphia)
    uv run python -m rrs.ingest --metro santa_barbara
    uv run python -m rrs.ingest --metro philadelphia --db data/yelp.duckdb
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import duckdb

from rrs.config import DB_PATH, DEFAULT_METRO, METROS, RAW_FILES, Metro, get_metro

# Explicit column schemas. Specifying columns makes the JSON reader extract only these
# keys (skipping the messy nested `attributes`/`hours` structs on businesses) and pins
# types so schema inference can't drift across the multi-GB files.
BUSINESS_COLS = {
    "business_id": "VARCHAR",
    "name": "VARCHAR",
    "address": "VARCHAR",
    "city": "VARCHAR",
    "state": "VARCHAR",
    "postal_code": "VARCHAR",
    "latitude": "DOUBLE",
    "longitude": "DOUBLE",
    "stars": "DOUBLE",
    "review_count": "BIGINT",
    "is_open": "BIGINT",
    "categories": "VARCHAR",
}

REVIEW_COLS = {
    "review_id": "VARCHAR",
    "user_id": "VARCHAR",
    "business_id": "VARCHAR",
    "stars": "DOUBLE",
    "useful": "BIGINT",
    "funny": "BIGINT",
    "cool": "BIGINT",
    "text": "VARCHAR",
    "date": "TIMESTAMP",
}

USER_COLS = {
    "user_id": "VARCHAR",
    "name": "VARCHAR",
    "review_count": "BIGINT",
    "yelping_since": "TIMESTAMP",
    "useful": "BIGINT",
    "funny": "BIGINT",
    "cool": "BIGINT",
    "elite": "VARCHAR",
    "friends": "VARCHAR",
    "fans": "BIGINT",
    "average_stars": "DOUBLE",
    "compliment_hot": "BIGINT",
    "compliment_more": "BIGINT",
    "compliment_profile": "BIGINT",
    "compliment_cute": "BIGINT",
    "compliment_list": "BIGINT",
    "compliment_note": "BIGINT",
    "compliment_plain": "BIGINT",
    "compliment_cool": "BIGINT",
    "compliment_funny": "BIGINT",
    "compliment_writer": "BIGINT",
    "compliment_photos": "BIGINT",
}

TIP_COLS = {
    "user_id": "VARCHAR",
    "business_id": "VARCHAR",
    "text": "VARCHAR",
    "date": "TIMESTAMP",
    "compliment_count": "BIGINT",
}

# checkin `date` is a single comma-separated string of timestamps; keep it raw and
# parse downstream when needed.
CHECKIN_COLS = {
    "business_id": "VARCHAR",
    "date": "VARCHAR",
}

TS_FORMAT = "%Y-%m-%d %H:%M:%S"


def _sql_struct(cols: dict[str, str]) -> str:
    """Render a column map as a DuckDB struct literal for read_json's `columns=`."""
    inner = ", ".join(f"'{name}': '{dtype}'" for name, dtype in cols.items())
    return "{" + inner + "}"


def _read_json(path: Path, cols: dict[str, str], *, with_timestamp: bool) -> str:
    """Build a read_json(...) table function call for a raw file."""
    literal = str(path).replace("'", "''")
    parts = [
        f"'{literal}'",
        "format = 'newline_delimited'",
        f"columns = {_sql_struct(cols)}",
    ]
    if with_timestamp:
        parts.append(f"timestampformat = '{TS_FORMAT}'")
    return "read_json(" + ", ".join(parts) + ")"


def _business_filter(metro: Metro) -> str:
    states = ", ".join(f"'{s}'" for s in metro.states)
    clause = f"state IN ({states})"
    if metro.cities:
        cities = ", ".join("'" + c.replace("'", "''") + "'" for c in metro.cities)
        clause += f" AND city IN ({cities})"
    return clause


def _step(con: duckdb.DuckDBPyConnection, label: str, sql: str, table: str) -> int:
    """Run a CREATE TABLE step, time it, and return the resulting row count."""
    t0 = time.perf_counter()
    print(f"  → {label} ...", flush=True)
    con.execute(f"DROP TABLE IF EXISTS {table}")
    con.execute(sql)
    n = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
    print(f"    {table}: {n:,} rows  ({time.perf_counter() - t0:.1f}s)", flush=True)
    return n


def ingest(metro: Metro, db_path: Path = DB_PATH) -> dict[str, int]:
    """Load one metro from the raw JSON dump into a fresh set of DuckDB tables."""
    for name, path in RAW_FILES.items():
        if not path.exists():
            raise SystemExit(f"Missing raw file for '{name}': {path}")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Ingesting metro '{metro.key}' ({metro.name}) → {db_path}")

    con = duckdb.connect(str(db_path))
    counts: dict[str, int] = {}
    try:
        con.execute("PRAGMA enable_progress_bar")

        biz_src = _read_json(RAW_FILES["business"], BUSINESS_COLS, with_timestamp=False)
        counts["businesses"] = _step(
            con,
            f"businesses where {_business_filter(metro)}",
            f"CREATE TABLE businesses AS SELECT * FROM {biz_src} WHERE {_business_filter(metro)}",
            "businesses",
        )
        if counts["businesses"] == 0:
            raise SystemExit(f"No businesses matched metro '{metro.key}'. Check the filter.")

        rev_src = _read_json(RAW_FILES["review"], REVIEW_COLS, with_timestamp=True)
        counts["reviews"] = _step(
            con,
            "reviews on those businesses (streams the 5 GB review file)",
            f"""CREATE TABLE reviews AS
                SELECT r.* FROM {rev_src} r
                WHERE r.business_id IN (SELECT business_id FROM businesses)""",
            "reviews",
        )

        usr_src = _read_json(RAW_FILES["user"], USER_COLS, with_timestamp=True)
        counts["users"] = _step(
            con,
            "users who wrote those reviews (streams the 3 GB user file)",
            f"""CREATE TABLE users AS
                SELECT u.* FROM {usr_src} u
                WHERE u.user_id IN (SELECT DISTINCT user_id FROM reviews)""",
            "users",
        )

        tip_src = _read_json(RAW_FILES["tip"], TIP_COLS, with_timestamp=True)
        counts["tips"] = _step(
            con,
            "tips on those businesses",
            f"""CREATE TABLE tips AS
                SELECT t.* FROM {tip_src} t
                WHERE t.business_id IN (SELECT business_id FROM businesses)""",
            "tips",
        )

        chk_src = _read_json(RAW_FILES["checkin"], CHECKIN_COLS, with_timestamp=False)
        counts["checkins"] = _step(
            con,
            "checkins on those businesses",
            f"""CREATE TABLE checkins AS
                SELECT c.* FROM {chk_src} c
                WHERE c.business_id IN (SELECT business_id FROM businesses)""",
            "checkins",
        )

        print("  → indexes ...", flush=True)
        con.execute("CREATE INDEX idx_reviews_business ON reviews(business_id)")
        con.execute("CREATE INDEX idx_reviews_user ON reviews(user_id)")
        con.execute("CREATE INDEX idx_reviews_date ON reviews(date)")
        con.execute("CREATE INDEX idx_tips_business ON tips(business_id)")

        # Stamp the DB with which metro it holds so downstream code can sanity-check.
        con.execute(
            "CREATE OR REPLACE TABLE meta (metro VARCHAR, name VARCHAR, loaded_at TIMESTAMP)"
        )
        con.execute(
            "INSERT INTO meta VALUES (?, ?, now())",
            [metro.key, metro.name],
        )
    finally:
        con.close()

    print("\nDone. Row counts:")
    for table, n in counts.items():
        print(f"  {table:12} {n:>12,}")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Load one Yelp metro into DuckDB.")
    parser.add_argument(
        "--metro",
        default=DEFAULT_METRO,
        choices=sorted(METROS),
        help=f"Metro to ingest (default: {DEFAULT_METRO})",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DB_PATH,
        help=f"Output DuckDB path (default: {DB_PATH})",
    )
    args = parser.parse_args()
    ingest(get_metro(args.metro), db_path=args.db)


if __name__ == "__main__":
    main()
