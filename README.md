# Real Rating Score (RRS)

ML-powered detection of suspicious/botted reviews on the [Yelp Open Dataset](https://www.yelp.com/dataset),
producing an adjusted "real" rating per business. See [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md)
for the full design and rationale.

## Status

| Phase | What | State |
|------|------|-------|
| 1 | Data foundation — JSON → DuckDB, EDA | ✅ built |
| 2 | Weak supervision labels (Snorkel) | ✅ built |
| 3 | Feature engineering (Polars) | ✅ built |
| 4 | Suspicion model (LightGBM + SHAP) | ✅ built |
| 5 | Real Rating Score aggregation | ⬜ next |
| 6 | FastAPI service | ⬜ |
| 7 | Next.js frontend | ⬜ |

**Current metro: Philadelphia.** The Yelp Open Dataset (Jan 2022) does **not** contain Los
Angeles; its largest metro is Philadelphia (the "LA" state code in the data is Louisiana /
New Orleans). Other metros are switchable via `--metro` — see `src/rrs/config.py`.

## Setup

Requires [`uv`](https://docs.astral.sh/uv/) and the raw dump unzipped at
`Yelp JSON/yelp_dataset/` (gitignored — academic-use license).

```bash
uv sync                      # core + dev deps (Python 3.12)
# later phases:
# uv sync --extra ml         # Snorkel, LightGBM, sentence-transformers, ...
# uv sync --extra api        # FastAPI
```

## Phase 1 — data foundation

```bash
uv run python -m rrs.ingest                 # load Philadelphia into data/yelp.duckdb
uv run python -m rrs.ingest --metro santa_barbara
uv run python -m rrs.eda                    # print distributions, save figures + report
```

`ingest` streams the 5 GB review and 3 GB user files, keeping only the rows belonging to
the target metro. `eda` writes `reports/eda_<metro>.md` and figures to `reports/figures/`.
The exploratory notebook lives at `notebooks/01_eda.ipynb`.

## Phase 2 — weak supervision labels

```bash
uv sync --extra ml                          # snorkel, sentence-transformers, lightgbm, ...
uv run python -m rrs.labeling.apply         # → labels/weak_labels.parquet + audit report
```

10 vectorized labeling functions (4 behavioral, 3 content, 3 account-quality) feed
Snorkel's `LabelModel`, which combines their noisy votes into `p_suspicious ∈ [0, 1]`
per review. The CLI also writes `reports/labels_<metro>.md` with per-LF coverage stats
and the top 100 highest-confidence suspicious reviews for manual audit.

On the 1.93M Philadelphia reviews the full pipeline runs in ~8 minutes; the
duplicate-text per-user TF-IDF is the dominant cost.

## Phase 3 — feature engineering

```bash
uv run python -m rrs.features.build               # → features/reviews.parquet + embeddings
uv run python -m rrs.features.build --skip-embeddings   # everything except MiniLM
uv run python -m rrs.features.build --sample 50000      # fast debug subset
```

Three feature families joined per review and written sorted by `business_id`:
**reviewer** (11 per-user aggregates: tenure, rating moments, social, posting-hour
entropy), **content** (lengths + caps/exclamation/first-person ratios + VADER + Flesch
+ MiniLM embedding + per-user / per-business max-cosine scalars), **context** (stars
deltas, prev-review lag, cumulative count, burst flag).

`features/reviews.parquet` is ~200 MB / 36 columns. The 384-dim MiniLM embeddings
themselves live separately at `features/embeddings.npy` (memmap-friendly, 2.96 GB)
keyed by `features/embeddings_index.parquet` so Phase 4 can choose to consume them
raw, PCA-reduce, or use only the derived similarity scalars.

End-to-end on Philadelphia (1.93M reviews, MPS GPU): ~55 min, dominated by the
MiniLM encode step at ~613 texts/sec.

## Phase 4 — suspicion model

```bash
uv run python -m rrs.modeling.train               # → models/ + reports/model_<metro>.md (~15 min)
uv run python -m rrs.modeling.train --n-trials 15 # lighter Optuna search
uv run python -m rrs.modeling.train --sample 80000 --no-shap   # fast debug
```

A LightGBM model is trained to predict the Snorkel `p_suspicious` soft label from the
Phase 3 features, using the `cross_entropy` objective (native [0, 1] targets). The
train/test split is **by time at 2020** — never random — and ~40 Optuna trials tune
against a held-out 2019 validation fold before the final model is refit on the full
pre-2020 period. Outputs: `models/lgbm_suspicion.txt`, a pickled SHAP `TreeExplainer`,
`models/predictions.parquet` (`review_id → p_fake` for every review, the Phase 5 input),
and a report with metrics, feature importances, a synthetic-injection probe, a held-out-
heuristic ablation, and the top-100 reviews by `p_fake` for manual audit.

Because there is no labeled ground truth, every metric is measured **against the
LabelModel's own output** — so a high AUC means the model faithfully reproduces and
generalizes the weak labels, not that it has detected verified fakes. Two known leakage
caveats (snapshot-relative per-user aggregates; the weak-label circularity) are spelled
out in the report. Treat `p_fake` as a calibrated second opinion, never a verdict.

## Layout

```
src/rrs/
├── config.py        # paths + metro definitions
├── ingest.py        # Phase 1: JSON → DuckDB
├── eda.py           # Phase 1: distributions / report
├── labeling/        # Phase 2
│   ├── constants.py # ABSTAIN/AUTHENTIC/SUSPICIOUS + thresholds
│   ├── enrich.py    # DuckDB query → per-review enriched DataFrame
│   ├── dup_text.py  # per-user TF-IDF near-duplicate similarity
│   ├── lfs.py       # 10 vectorized labeling functions
│   └── apply.py     # combine LFs → LabelModel → weak_labels.parquet
├── features/        # Phase 3
│   ├── reviewer.py  # per-user aggregates (rating moments, social, hour entropy)
│   ├── content.py   # text features + VADER + Flesch (multiprocessed)
│   ├── context.py   # per-review deltas, lag, cumulative count, burst flag
│   ├── embeddings.py # MiniLM encode + per-key max-cosine scalars
│   └── build.py     # orchestrator → features/reviews.parquet
├── modeling/        # Phase 4
│   ├── dataset.py   # join features+labels, leakage-aware feature selection, time split
│   └── train.py     # Optuna-tuned LightGBM + eval + validation + SHAP
└── api/             # Phase 6
data/                # gitignored: yelp.duckdb
labels/              # gitignored: weak_labels.parquet
features/            # gitignored: reviews.parquet + embeddings.npy
models/              # gitignored: lgbm_suspicion.txt + shap_explainer.pkl + predictions.parquet
notebooks/           # 01_eda.ipynb
reports/             # EDA + labels audit + model report + figures
tests/
```

## License

Code: MIT. Data: see the [Yelp Dataset License](https://www.yelp.com/dataset) (academic/personal use).
