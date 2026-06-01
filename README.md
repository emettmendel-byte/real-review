# Real Rating Score (RRS)

ML-powered detection of suspicious/botted reviews on the [Yelp Open Dataset](https://www.yelp.com/dataset),
producing an adjusted "real" rating per business. See [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md)
for the full design and rationale.

## Status

| Phase | What | State |
|------|------|-------|
| 1 | Data foundation — JSON → DuckDB, EDA | ✅ built |
| 2 | Weak supervision labels (Snorkel) | ✅ built |
| 3 | Feature engineering (Polars) | ⬜ next |
| 4 | Suspicion model (LightGBM + SHAP) | ⬜ |
| 5 | Real Rating Score aggregation | ⬜ |
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
├── modeling/        # Phase 4
└── api/             # Phase 6
data/                # gitignored: yelp.duckdb
labels/              # gitignored: weak_labels.parquet
notebooks/           # 01_eda.ipynb
reports/             # EDA + labels audit + figures
tests/
```

## License

Code: MIT. Data: see the [Yelp Dataset License](https://www.yelp.com/dataset) (academic/personal use).
