"""Phase 1 EDA — distributions and sanity checks on the ingested metro.

Importable query functions (each returns a pandas DataFrame) used by the notebook,
plus a `main()` that prints a text report, saves figures, and writes a markdown summary.

    uv run python -m rrs.eda                  # report for data/yelp.duckdb
    uv run python -m rrs.eda --db data/yelp.duckdb
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import pandas as pd

from rrs.config import DB_PATH, REPORTS_DIR


def connect(db_path: Path = DB_PATH) -> duckdb.DuckDBPyConnection:
    if not db_path.exists():
        raise SystemExit(f"No DuckDB file at {db_path}. Run `python -m rrs.ingest` first.")
    return duckdb.connect(str(db_path), read_only=True)


def metro_label(con: duckdb.DuckDBPyConnection) -> str:
    try:
        row = con.execute("SELECT name FROM meta LIMIT 1").fetchone()
        return row[0] if row else "unknown"
    except duckdb.Error:
        return "unknown"


def overview(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    rows = []
    for table in ("businesses", "reviews", "users", "tips", "checkins"):
        n = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        rows.append({"table": table, "rows": n})
    return pd.DataFrame(rows)


def geo_breakdown(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return con.execute(
        """SELECT state, count(*) AS businesses,
                  sum(review_count) AS yelp_review_count
           FROM businesses GROUP BY state ORDER BY businesses DESC"""
    ).df()


def top_cities(con: duckdb.DuckDBPyConnection, limit: int = 15) -> pd.DataFrame:
    return con.execute(
        f"""SELECT city, state, count(*) AS businesses
            FROM businesses GROUP BY city, state
            ORDER BY businesses DESC LIMIT {limit}"""
    ).df()


def review_date_range(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return con.execute(
        """SELECT min(date) AS first_review, max(date) AS last_review,
                  count(*) AS n_reviews FROM reviews"""
    ).df()


def reviews_per_user(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Distribution of how many reviews each user wrote *within this metro*."""
    return con.execute(
        """WITH per_user AS (
               SELECT user_id, count(*) AS n FROM reviews GROUP BY user_id
           )
           SELECT
               count(*)                               AS n_users,
               min(n)                                 AS min,
               quantile_cont(n, 0.50)                 AS p50,
               quantile_cont(n, 0.90)                 AS p90,
               quantile_cont(n, 0.99)                 AS p99,
               max(n)                                 AS max,
               avg(n)                                 AS mean,
               sum(CASE WHEN n = 1 THEN 1 ELSE 0 END) AS one_review_users
           FROM per_user"""
    ).df()


def reviews_per_business(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return con.execute(
        """WITH per_biz AS (
               SELECT business_id, count(*) AS n FROM reviews GROUP BY business_id
           )
           SELECT count(*) AS n_businesses, min(n) AS min,
                  quantile_cont(n, 0.50) AS p50, quantile_cont(n, 0.90) AS p90,
                  quantile_cont(n, 0.99) AS p99, max(n) AS max, avg(n) AS mean
           FROM per_biz"""
    ).df()


def rating_distribution(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return con.execute(
        """SELECT stars, count(*) AS n,
                  round(100.0 * count(*) / sum(count(*)) OVER (), 1) AS pct
           FROM reviews GROUP BY stars ORDER BY stars"""
    ).df()


def reviews_per_year(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return con.execute(
        """SELECT year(date) AS yr, count(*) AS n
           FROM reviews GROUP BY yr ORDER BY yr"""
    ).df()


def review_length(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return con.execute(
        """WITH lens AS (SELECT length(text) AS c FROM reviews)
           SELECT min(c) AS min, quantile_cont(c, 0.50) AS p50,
                  quantile_cont(c, 0.90) AS p90, quantile_cont(c, 0.99) AS p99,
                  max(c) AS max, avg(c) AS mean FROM lens"""
    ).df()


def account_age_vs_reviews(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Lifetime review_count bucketed by account age — a first look at the kind of
    signal weak-supervision LFs will lean on (new accounts, one-shot reviewers)."""
    return con.execute(
        """WITH u AS (
               SELECT review_count,
                      date_diff('day', yelping_since, DATE '2022-01-19') AS age_days
               FROM users
           )
           SELECT
               CASE
                   WHEN age_days < 30   THEN '0  <30d'
                   WHEN age_days < 180  THEN '1  30-180d'
                   WHEN age_days < 365  THEN '2  180-365d'
                   WHEN age_days < 1095 THEN '3  1-3y'
                   ELSE '4  >3y'
               END AS account_age,
               count(*)            AS n_users,
               avg(review_count)   AS avg_lifetime_reviews,
               median(review_count) AS median_lifetime_reviews
           FROM u GROUP BY account_age ORDER BY account_age"""
    ).df()


def _save_figures(con: duckdb.DuckDBPyConnection, out_dir: Path) -> list[Path]:
    """Save core distribution plots; no-op-ish if matplotlib is unavailable."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not installed — skipping figures)")
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    ratings = rating_distribution(con)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(ratings["stars"].astype(str), ratings["n"], color="#4C78A8")
    ax.set(title="Review rating distribution", xlabel="stars", ylabel="reviews")
    fig.tight_layout()
    p = out_dir / "rating_distribution.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    saved.append(p)

    yearly = reviews_per_year(con)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(yearly["yr"], yearly["n"], marker="o", color="#E45756")
    ax.set(title="Reviews per year", xlabel="year", ylabel="reviews")
    fig.tight_layout()
    p = out_dir / "reviews_per_year.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    saved.append(p)

    rpu = con.execute(
        "SELECT count(*) AS n FROM reviews GROUP BY user_id"
    ).df()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(rpu["n"].clip(upper=30), bins=30, color="#54A24B")
    ax.set(title="Reviews per user (clipped at 30)", xlabel="reviews by user", ylabel="users")
    ax.set_yscale("log")
    fig.tight_layout()
    p = out_dir / "reviews_per_user.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    saved.append(p)

    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 EDA report for an ingested metro.")
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--no-figures", action="store_true", help="skip saving PNG figures")
    args = parser.parse_args()

    con = connect(args.db)
    label = metro_label(con)

    sections: list[tuple[str, pd.DataFrame]] = [
        ("Overview (table row counts)", overview(con)),
        ("Businesses by state", geo_breakdown(con)),
        ("Top cities", top_cities(con)),
        ("Review date range", review_date_range(con)),
        ("Reviews per user (within metro)", reviews_per_user(con)),
        ("Reviews per business", reviews_per_business(con)),
        ("Rating distribution", rating_distribution(con)),
        ("Reviews per year", reviews_per_year(con)),
        ("Review length (chars)", review_length(con)),
        ("Account age vs lifetime reviews", account_age_vs_reviews(con)),
    ]

    print(f"\n{'=' * 64}\nEDA — {label}\n{'=' * 64}")
    for title, df in sections:
        print(f"\n## {title}")
        print(df.to_string(index=False))

    figures: list[Path] = []
    if not args.no_figures:
        print("\n## Figures")
        figures = _save_figures(con, REPORTS_DIR / "figures")
        for p in figures:
            print(f"  saved {p.relative_to(REPORTS_DIR.parent)}")

    # Markdown summary for the check-in / repo.
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    md_path = REPORTS_DIR / f"eda_{label.lower().replace(' ', '_')}.md"
    lines = [f"# EDA — {label}", ""]
    for title, df in sections:
        lines += [f"## {title}", "", df.to_markdown(index=False), ""]
    if figures:
        lines += ["## Figures", ""]
        lines += [f"![{p.stem}](figures/{p.name})" for p in figures]
        lines += [""]
    md_path.write_text("\n".join(lines))
    print(f"\nWrote {md_path}")
    con.close()


if __name__ == "__main__":
    main()
