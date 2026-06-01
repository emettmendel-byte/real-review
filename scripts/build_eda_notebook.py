"""Generate notebooks/01_eda.ipynb from the rrs.eda query functions.

Keeping the notebook generated (rather than hand-edited JSON) means the exploratory
notebook and the `rrs.eda` module never drift apart. Run after editing rrs.eda:

    uv run python scripts/build_eda_notebook.py
    uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "01_eda.ipynb"

nb = nbf.v4.new_notebook()
cells: list = []

cells.append(
    nbf.v4.new_markdown_cell(
        "# 01 — EDA (Phase 1)\n\n"
        "Distributions and sanity checks on the ingested metro. Run `python -m rrs.ingest` "
        "first to build `data/yelp.duckdb`. Query logic lives in `rrs.eda`; this notebook "
        "just displays it."
    )
)

cells.append(
    nbf.v4.new_code_cell(
        "import sys; sys.path.insert(0, '../src')\n"
        "import matplotlib.pyplot as plt\n"
        "from rrs import eda\n\n"
        "con = eda.connect()\n"
        "label = eda.metro_label(con)\n"
        "print('Metro:', label)"
    )
)

cells.append(nbf.v4.new_markdown_cell("## Overview & geography"))
cells.append(nbf.v4.new_code_cell("eda.overview(con)"))
cells.append(nbf.v4.new_code_cell("eda.geo_breakdown(con)"))
cells.append(nbf.v4.new_code_cell("eda.top_cities(con)"))
cells.append(nbf.v4.new_code_cell("eda.review_date_range(con)"))

cells.append(nbf.v4.new_markdown_cell("## Reviews per user / per business"))
cells.append(nbf.v4.new_code_cell("eda.reviews_per_user(con)"))
cells.append(nbf.v4.new_code_cell("eda.reviews_per_business(con)"))
cells.append(
    nbf.v4.new_code_cell(
        "rpu = con.execute('SELECT count(*) AS n FROM reviews GROUP BY user_id').df()\n"
        "ax = rpu['n'].clip(upper=30).plot.hist(bins=30, logy=True, color='#54A24B')\n"
        "ax.set(title='Reviews per user (clipped at 30)', xlabel='reviews by user'); plt.show()"
    )
)

cells.append(nbf.v4.new_markdown_cell("## Ratings & temporal patterns"))
cells.append(nbf.v4.new_code_cell("eda.rating_distribution(con)"))
cells.append(
    nbf.v4.new_code_cell(
        "r = eda.rating_distribution(con)\n"
        "ax = r.plot.bar(x='stars', y='n', legend=False, color='#4C78A8')\n"
        "ax.set(title='Rating distribution', ylabel='reviews'); plt.show()"
    )
)
cells.append(
    nbf.v4.new_code_cell(
        "y = eda.reviews_per_year(con)\n"
        "ax = y.plot(x='yr', y='n', marker='o', legend=False, color='#E45756')\n"
        "ax.set(title='Reviews per year', ylabel='reviews'); plt.show()"
    )
)

cells.append(nbf.v4.new_markdown_cell("## Review length & account quality"))
cells.append(nbf.v4.new_code_cell("eda.review_length(con)"))
cells.append(nbf.v4.new_code_cell("eda.account_age_vs_reviews(con)"))

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}

OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(f"Wrote {OUT}")
