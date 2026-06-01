# CLAUDE.md

Guidance for working in this repo. For the full design and rationale see
[`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md); for setup and phase status see
[`README.md`](README.md). This file is the short operational layer — don't duplicate
those; link to them.

## What this is

**Real Rating Score (RRS)** — ML detection of suspicious/botted reviews on the Yelp Open
Dataset, producing an adjusted per-business rating. Seven phases: ingest → weak labels →
features → LightGBM → RRS scoring → FastAPI → Next.js (see the plan).

## Critical facts (read before touching data)

- **No Los Angeles in the dataset.** The Yelp Open Dataset (Jan 2022) covers a fixed set
  of metros. The `LA` state code is **Louisiana** (New Orleans), not Los Angeles. The only
  real California metro is Santa Barbara. We target **Philadelphia** — see the metro
  registry in `src/rrs/config.py`. Don't add a `los_angeles` metro; it won't match rows.
- **Philadelphia = states {PA, NJ, DE}** (metro incl. suburbs) → 44.8K businesses / 1.93M
  reviews. This is intentionally larger than the plan's "city-only ~900K" estimate. To
  match the plan's footprint, narrow `Metro.cities` to `("Philadelphia",)` in config.
- **Raw data is gitignored** at `Yelp JSON/yelp_dataset/*.json` (5 GB review, 3 GB user;
  academic-use license — never commit it).
- **`data/yelp.duckdb` is regenerable** and gitignored. If it's missing or schema changed,
  rebuild with `rrs.ingest`. There's a `meta` table stamping which metro it holds.
- 7 reviews reference users absent from the user file — use LEFT JOINs to `users`.

## Environment & commands

`uv` + Python **3.12** (pinned in `.python-version`; the system Python 3.14 is too new for
the ML wheels). Core/dev deps install fast; heavy stacks are behind extras.

```bash
uv sync                         # core + dev (duckdb, polars, jupyter, ruff, pytest)
uv sync --extra ml              # Phase 2+: snorkel, lightgbm, sentence-transformers, shap...
uv sync --extra api             # Phase 6: fastapi, uvicorn

uv run python -m rrs.ingest                    # load Philadelphia → data/yelp.duckdb (~33s)
uv run python -m rrs.ingest --metro santa_barbara
uv run python -m rrs.eda                       # print distributions, write reports/ + figures
uv run python -m rrs.labeling.apply            # → labels/weak_labels.parquet + audit (~8 min)
uv run python scripts/build_eda_notebook.py    # regenerate notebooks/01_eda.ipynb from rrs.eda
uv run pytest -q                               # tests (pure-function, no DB needed)
uv run ruff check src tests scripts            # lint (line-length 100)
```

> **macOS only:** `brew install libomp` once before using `--extra ml`, or LightGBM
> will fail at import with `Library not loaded: @rpath/libomp.dylib`.

## Layout

Source is a `src/rrs/` package (see `README.md` for the tree). Key modules:

- `config.py` — paths + `METROS` registry + `get_metro()`. Single source of truth for
  where raw files and the DB live.
- `ingest.py` — Phase 1 JSON→DuckDB. Explicit column schemas (`*_COLS` dicts) pin types and
  skip nested `attributes`/`hours`. SQL is built by small helpers (`_read_json`,
  `_business_filter`) that are unit-tested.
- `eda.py` — Phase 1 query functions (each returns a DataFrame) + `main()` report writer.
- `labeling/` — Phase 2 weak supervision. `enrich.py` runs one DuckDB query for
  per-review burst/gap/social context; `dup_text.py` does per-user TF-IDF near-duplicate
  detection; `lfs.py` holds the 10 vectorized Polars LFs (each returns Int8 of
  ABSTAIN/AUTHENTIC/SUSPICIOUS); `apply.py` stacks them into a label matrix, fits
  Snorkel's `LabelModel`, and writes `labels/weak_labels.parquet` + an audit report.
- `features/`, `modeling/`, `api/` — empty packages for Phases 3–6.

DuckDB tables: `businesses`, `reviews`, `users`, `tips`, `checkins`, `meta`.

## Conventions

- **EDA logic lives in `rrs.eda`, not the notebook.** The notebook is generated from it via
  `scripts/build_eda_notebook.py` so the two never drift. Edit the module, regenerate,
  re-execute with `jupyter nbconvert --to notebook --execute --inplace`.
- New query/analysis logic → an importable function in a module, displayed by a thin
  notebook cell. Keep notebooks as views, not logic.
- Time-based split for modeling (train pre-2020, test 2020+) — never random; random splits
  leak signal. ~1.66M train / ~266K test in Philadelphia.
- Treat `p_fake` as a probability/second opinion, never a verdict (see the plan's "Honest
  limitations" and frontend tone notes).
- `ruff` line length 100; rules in `pyproject.toml`. Tests must not require the multi-GB
  files or a built DB.
- **LFs must vote on both polarities** ([0, 1] in `LFAnalysis`). One-sided LFs degrade
  Snorkel's `LabelModel` to majority vote. Every LF in `lfs.py` therefore has both a
  SUSPICIOUS branch and an AUTHENTIC branch — change thresholds carefully.
- **Snorkel 0.10 API quirk:** `LabelModel` is at `snorkel.labeling.model`, not
  `snorkel.labeling` (the docs you'll find online describe the older 0.9 layout).

## Status

Phases 1 (data foundation) and 2 (weak supervision) are done. On Philadelphia: 1.93M
reviews scored, 21% flagged at `p_suspicious > 0.5`, distribution is healthily bimodal
(P50≈0.05, P90≈0.93). Top-100 audit in `reports/labels_philadelphia.md`. Next: Phase 3
feature engineering (the reviewer/content/context families in the plan). See `README.md`
for the full status table.
