# Real Rating Score (RRS)

> ML-powered detection of suspicious/botted reviews, producing an adjusted "real" rating for businesses. Built on the Yelp Open Dataset.

## What this is

Online review systems are gamed. Restaurants buy 5★ reviews, competitors plant 1★ reviews, and platforms' own filters are opaque. **RRS** computes a transparent, adjusted rating score by estimating the probability that each review is inauthentic and down-weighting it accordingly.

**Output per business**: an adjusted rating, a confidence interval, a fake-review percentage, and per-review explanations.

---

## Technical decisions (and why)

These are the opinionated choices that shape the rest of the project. Each one can be revisited later, but starting with a fixed stack avoids analysis paralysis.

| Decision | Choice | Rationale |
|---|---|---|
| Data storage | **DuckDB** | Queries Yelp's JSON files directly. No ETL step, no separate database server, single-file portability. Postgres is overkill until we serve traffic. |
| Scope | **One metro at a time** (start: Philadelphia) | Yelp's full dataset is ~7M reviews. Single-metro keeps iteration loops under 5 minutes and graph structure tractable. |
| Labeling | **Weak supervision via Snorkel** | No ground-truth fake labels exist. Snorkel lets us encode noisy heuristics as labeling functions and combine them probabilistically — beats hand-tuned rules and is auditable. |
| Primary model | **LightGBM** | Faster than XGBoost on tabular data, native categorical handling, excellent SHAP integration. Transformers are overkill for the v1 feature set. |
| Text embeddings | **`all-MiniLM-L6-v2`** (sentence-transformers) | 384-dim, runs on CPU, good enough for similarity-based features. Upgrade to a fine-tuned model only if text signal dominates. |
| Graph features | **Deferred to v2** | Bipartite reviewer-business GNNs help with coordinated fraud rings but add significant complexity. Ship tabular v1 first; node2vec embeddings come after. |
| Score aggregation | **Bayesian shrinkage + weighted mean** | A business with 3 reviews shouldn't swing wildly. Shrink toward the global mean with a prior weight of ~10 reviews. |
| Backend | **FastAPI + DuckDB** | Async, typed, fast. DuckDB serves analytical queries directly — no separate OLAP layer needed at this scale. |
| Frontend | **Next.js + Tailwind + shadcn/ui** | Standard, fast to build, server components keep the data-heavy pages snappy. |
| Deployment (v1) | **Single VPS / Fly.io** | One container, no orchestration. Re-score job runs as a cron. Scale later if needed. |

---

## Architecture

```
┌──────────────────┐
│  Yelp JSON dump  │
└────────┬─────────┘
         │ (one-time load)
         ▼
┌──────────────────┐     ┌────────────────────┐
│  DuckDB (.duckdb │◄────┤ Feature pipeline   │
│  file)           │     │ (Python, Polars)   │
└────────┬─────────┘     └────────────────────┘
         │
         ▼
┌──────────────────┐     ┌────────────────────┐
│  Snorkel weak    │────►│  LightGBM model    │
│  labels          │     │  (suspicion p_fake)│
└──────────────────┘     └─────────┬──────────┘
                                   │
                                   ▼
                         ┌────────────────────┐
                         │  RRS aggregator    │
                         │  (per-business)    │
                         └─────────┬──────────┘
                                   │
                                   ▼
                         ┌────────────────────┐
                         │  FastAPI service   │
                         └─────────┬──────────┘
                                   │
                                   ▼
                         ┌────────────────────┐
                         │  Next.js frontend  │
                         └────────────────────┘
```

---

## Phase 1 — Data foundation

**Goal**: queryable dataset, basic understanding of distributions.

- Download the Yelp Open Dataset (~10GB, JSON). Note the academic-use license.
- Load JSON files into DuckDB tables: `businesses`, `reviews`, `users`, `tips`, `checkins`.
- Filter to a single metro (Philadelphia: ~30K businesses, ~900K reviews — large enough to be interesting, small enough to iterate fast).
- EDA notebook covering: reviews-per-user distribution, reviews-per-business distribution, temporal patterns, account-age vs review-count, rating distributions, review-length distributions.

**Deliverable**: `data/yelp.duckdb` and `notebooks/01_eda.ipynb`.

---

## Phase 2 — Weak supervision labels

**Goal**: a per-review noisy label `y ∈ {SUSPICIOUS, AUTHENTIC, ABSTAIN}` from many labeling functions, combined into a probabilistic label.

Labeling functions to implement (each returns one of the three values):

**Behavioral**
- `lf_burst`: review posted during an abnormal spike for this business (>3σ over rolling 7-day window)
- `lf_one_shot_extreme`: user has ≤2 lifetime reviews and this one is 1★ or 5★
- `lf_rating_deviation`: review stars deviate >2 from business mean and reviewer's history
- `lf_temporal_regularity`: user posts at suspiciously regular intervals (low variance in inter-review gaps)

**Content**
- `lf_duplicate_text`: cosine similarity >0.9 with another review by same user
- `lf_template_text`: high similarity to known template patterns
- `lf_extreme_brevity`: <20 chars on a 5★ review

**Account quality**
- `lf_new_account`: account <30 days old at review time
- `lf_no_social`: zero friends, zero fans, zero photos
- `lf_account_burst`: account posted >5 reviews in first week

Run Snorkel's `LabelModel` to combine these into `p_suspicious` per review. Validate by spot-checking the top 100 highest-confidence positives manually.

**Deliverable**: `labels/weak_labels.parquet` with one row per review.

---

## Phase 3 — Feature engineering

Three feature families, computed with Polars (faster than pandas for this size).

**Reviewer features** (per user, joined to each review)
- account_age_days, total_reviews, reviews_per_month, rating_variance, rating_skew, friend_count, fan_count, photo_count, fraction_extreme_ratings, avg_review_length, posting_hour_entropy

**Review content features**
- char_length, word_count, sentence_count, readability (Flesch), VADER sentiment, exclamation_ratio, caps_ratio, first_person_ratio, embedding (384-dim MiniLM), max_similarity_to_user_history, max_similarity_to_business_reviews

**Context features**
- stars_delta_from_business_mean, stars_delta_from_user_mean, hours_since_prev_review_on_business, is_in_burst_window, business_review_count_at_time

Persist as `features/reviews.parquet` partitioned by business.

---

## Phase 4 — Suspicion model

**Training**:
- Targets from Snorkel `p_suspicious` (soft labels — LightGBM handles these directly via custom objective or threshold at 0.5 for binary)
- Train/test split by **time**, not random — train on reviews before 2020, test after. Random splits leak signal.
- 5-fold CV within training period for hyperparameter tuning (Optuna, ~50 trials)

**Validation** (since labels are noisy):
1. **Synthetic injection**: generate fake reviews (paraphrase real ones, mass-post under burner accounts) and measure recall
2. **Manual audit**: precision@100 on highest-scored reviews, scored by you
3. **Held-out heuristics**: train without one labeling function family, see if the model recovers it

**Output**: `p_fake ∈ [0, 1]` per review, with SHAP values cached for top contributing features.

**Deliverable**: `models/lgbm_suspicion.txt` + `models/shap_explainer.pkl`.

---

## Phase 5 — Real Rating Score

Per business:

```python
# Weighted by authenticity
trust_weight_i = 1 - p_fake_i
weighted_sum = Σ (stars_i × trust_weight_i)
weight_total = Σ trust_weight_i

# Bayesian shrinkage toward global mean
PRIOR_WEIGHT = 10
GLOBAL_MEAN = 3.7  # computed from authentic-only reviews
rrs = (weighted_sum + PRIOR_WEIGHT × GLOBAL_MEAN) / (weight_total + PRIOR_WEIGHT)

# Confidence: Wilson interval based on effective sample size
n_eff = weight_total
ci_low, ci_high = wilson_interval(rrs, n_eff)

# Transparency metrics
pct_flagged = mean(p_fake_i > 0.5)
n_flagged = sum(p_fake_i > 0.5)
```

Surface all of these in the API — never just the bare RRS number.

---

## Phase 6 — API

FastAPI service, three endpoints for v1:

```
GET /businesses/search?q={query}&city={city}
GET /businesses/{business_id}
  → {
      name, address, yelp_rating, yelp_review_count,
      rrs, rrs_ci_low, rrs_ci_high,
      pct_flagged, n_authentic_reviews
    }
GET /businesses/{business_id}/reviews?include_flags=true
  → [{ review_id, stars, text, date, p_fake, top_signals: [...] }]
```

The `top_signals` field returns the 3 highest SHAP contributors as human-readable strings ("Account created 4 days before review", "Text 94% similar to another review by same user", etc.). This is the trust-building differentiator.

---

## Phase 7 — Frontend

Next.js app, three pages:

- `/` — search businesses by name/city
- `/business/[id]` — Yelp rating vs RRS side-by-side, % flagged, distribution chart of `p_fake` across reviews
- `/business/[id]/reviews` — review list with toggle for "show flagged" and per-review explainer

Tone: this is a **second opinion**, not a verdict. Flagged reviews are dimmed but visible, with an explanation. No accusations against specific users — show signals, not labels.

---

## Repo layout

```
rrs/
├── data/                    # gitignored, contains yelp.duckdb
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_label_validation.ipynb
│   └── 03_model_eval.ipynb
├── src/rrs/
│   ├── ingest.py            # JSON → DuckDB
│   ├── features/            # Polars feature pipelines
│   ├── labeling/            # Snorkel LFs and combiner
│   ├── modeling/            # LightGBM train/predict
│   ├── scoring.py           # RRS aggregation
│   └── api/                 # FastAPI app
├── frontend/                # Next.js
├── tests/
├── pyproject.toml           # uv for env management
└── README.md
```

---

## Roadmap

**v1 (5 weeks)**: tabular features, LightGBM, RRS endpoint, minimal frontend, Philadelphia only.

**v2**: graph features (reviewer-business bipartite, node2vec embeddings), expanded to all metros, model retraining pipeline.

**v3**: transformer-based text model fine-tuned on Snorkel labels, active learning loop where manual audits feed back into labeling functions, multi-platform support (Google Maps, TripAdvisor via their respective APIs/datasets).

---

## Honest limitations

- **No ground truth.** Every metric is relative to noisy weak labels. Treat all precision/recall numbers as directional.
- **Dataset is a snapshot** (~2022). This is a research/portfolio project, not a live fraud monitor.
- **False positives matter.** Real people post short, enthusiastic 5★ reviews after one visit. The UI must reflect uncertainty — never present `p_fake` as a verdict.
- **Yelp's filter exists.** Many obvious fakes are already filtered out of the open dataset, which makes our positive class scarcer than in the wild. Synthetic injection during eval partially mitigates this.
- **License.** Yelp Open Dataset is for academic/personal use. Commercial deployment requires a different data source.

---

## License

Code: MIT. Data: see [Yelp Dataset License](https://www.yelp.com/dataset).
