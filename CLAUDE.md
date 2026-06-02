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
uv run python -m rrs.features.build            # → features/reviews.parquet + embeddings (~55 min)
uv run python -m rrs.modeling.train            # → models/ + reports/model_<metro>.md (~15 min)
uv run python -m rrs.scoring                   # → models/rrs_scores.parquet + report (~5s)
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
- `features/` — Phase 3 feature engineering. `reviewer.py` per-user aggregates
  (rating moments, social, posting-hour entropy); `content.py` polars text features
  + multiprocessed VADER + a self-contained Flesch implementation (no NLTK); `context.py`
  per-review deltas, prev-review lag, cumulative count, and burst flag; `embeddings.py`
  MiniLM encode (uses MPS on Apple Silicon) + per-user/per-business max-cosine scalars;
  `build.py` orchestrator. Output: `features/reviews.parquet` (36 cols, sorted by
  business_id) + `embeddings.npy` (memmap-friendly) + `embeddings_index.parquet`.
- `modeling/` — Phase 4 suspicion model. `dataset.py` joins features to the Phase 2
  `p_suspicious` soft label, does leakage-aware feature selection (drops
  `account_age_days_snapshot`, keeps the point-in-time `account_age_days_at_review`), and
  the time split at 2020; `train.py` runs the Optuna search (single 2019 validation fold),
  refits on full pre-2020, evaluates on 2020+, runs the validation suite (synthetic
  injection, duplicate-text ablation, top-100 audit), and writes `models/` + the report.
- `scoring.py` — Phase 5 RRS aggregation. `wilson_interval` (proportion CI),
  `global_trust_weighted_mean` (μ), `score_businesses` (trust-weighted mean +
  Bayesian shrinkage toward μ, K=`PRIOR_WEIGHT`=10, Wilson CI on the [0,1]-rescaled
  rating). Output: `models/rrs_scores.parquet` (per business) — the Phase 6 input.
- `api/` — Phase 6 FastAPI service. `data.py` caches the booster, SHAP explainer,
  `predictions.parquet` and `rrs_scores.parquet` once at startup (lru_cache/lifespan) and
  opens DuckDB **read-only**; `signals.py` is the pure SHAP→plain-English mapping (one
  renderer per model feature, positive log-odds = pushes `p_fake` up, top-3 incriminating
  signals, never names a user); `app.py` is the FastAPI `app` with the three
  `/businesses` endpoints + `/health`. Runtime needs **both** extras:
  `uv sync --extra ml --extra api`; run with
  `PYTHONPATH=src uv run uvicorn rrs.api.app:app`.

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
- **Phase 4 trains on soft labels with `objective="cross_entropy"`** (LightGBM's native
  [0,1]-target objective) — *not* `binary`, which expects 0/1. Every model metric (AUC, AP,
  precision@k) is computed against the LabelModel's *own* binarized output, so a near-1.0
  AUC reflects faithful reproduction of the weak labels, not ground-truth fake detection.
  Keep that framing in any report or UI copy.
- **`feature_pre_filter=False` is required for the Optuna search.** Trials reuse one
  `lgb.Dataset` while varying `min_child_samples`; with pre-filtering on, lowering it below
  the first trial's value is a fatal error. It's set in `train._base_params()`.
- `ruff` line length 100; rules in `pyproject.toml`. Tests must not require the multi-GB
  files or a built DB.
- **LFs must vote on both polarities** ([0, 1] in `LFAnalysis`). One-sided LFs degrade
  Snorkel's `LabelModel` to majority vote. Every LF in `lfs.py` therefore has both a
  SUSPICIOUS branch and an AUTHENTIC branch — change thresholds carefully.
- **Snorkel 0.10 API quirk:** `LabelModel` is at `snorkel.labeling.model`, not
  `snorkel.labeling` (the docs you'll find online describe the older 0.9 layout).
- **Don't use `textstat`'s Flesch** — it transitively imports NLTK's CMU dictionary,
  which fails to load inside multiprocessing workers. `features/content.py` ships a
  self-contained Flesch implementation (vowel-group syllable count + silent-e).
- **MPS on Apple Silicon gives ~1.5× over CPU for MiniLM** (~680 vs ~470 texts/sec),
  not the 10× you'd hope. The model is tiny enough that tokenization (CPU) is a big
  chunk of the wall-clock. Don't crank batch size past 128 — larger batches stall on
  this hardware in ways that don't show up in CPU utilization.

## Status

Phases 1–4 are done. Philadelphia: 1.93M reviews fully featured (`features/reviews.parquet`
36 cols + `embeddings.npy` 2.96 GB), weakly labeled (`labels/weak_labels.parquet`), and
scored by the Phase 4 LightGBM (`models/lgbm_suspicion.txt`, `models/predictions.parquet`).
Test (2020+) AUC ≈ 0.999 / AP ≈ 0.998 **against the LabelModel's own output** — i.e. the
model faithfully reproduces the weak labels, not verified fakes. Gain is dominated by the
per-user aggregates (`total_reviews` ≈0.65, `friend_count` ≈0.17), which is also where the
documented temporal leakage lives; the embedding-similarity scalars get ~0 gain and the
duplicate-text ablation shows that heuristic is *not* reproduced. Phase 5 aggregates
`p_fake` into the per-business RRS (`models/rrs_scores.parquet`, 44,840 businesses;
μ=3.747, mean |RRS−naive|≈0.44 stars; biggest downward moves are small all-5★ shops with
~75% flagged). Phase 6 serves it all over FastAPI (`/businesses/search`, `/businesses/{id}`,
`/businesses/{id}/reviews` with per-review SHAP `top_signals`). Next: Phase 7 — Next.js
frontend in `frontend/`. See `README.md` for the full status table and `reports/` for
metrics + limitations.
