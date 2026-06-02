# RRS Frontend (Phase 7)

A thin Next.js (App Router) + TypeScript + Tailwind client over the Real Rating Score
FastAPI backend (Phase 6). It shows, for Philadelphia-area businesses, the **Yelp rating
side by side with the Real Rating Score**, a distribution of per-review fake-probability,
and a review list where likely-suspicious reviews are **dimmed but never hidden**, each
with the signals that drew the model's attention.

This is a **second opinion, not a verdict.** See the disclaimer in the app footer.

## Pages

- `/` — search businesses by name (+ optional city); results show Yelp vs RRS.
- `/business/[id]` — detail: Yelp vs RRS with its confidence range, transparency counts
  (`% flagged`, flagged/authentic/total), and a histogram of `p_fake` across reviews.
- `/business/[id]/reviews` — review list with a "show flagged reviews" toggle, per-review
  signal explainers, and a "load more" pager.

## Prerequisites

- Node 24 / npm 11 (no pnpm/yarn).
- The RRS backend running and reachable (default `http://localhost:8011`).

## Run the backend (from the repo root)

```bash
uv sync --extra ml --extra api
PYTHONPATH=src uv run uvicorn rrs.api.app:app --port 8011
```

> macOS: `brew install libomp` once first, or LightGBM fails to import.

Wait ~6s for the model/SHAP explainer to load, then `GET /health` returns
`{"status":"ok","metro":"philadelphia"}`.

## Run the frontend

```bash
cd frontend
npm install
cp .env.local.example .env.local   # optional; only needed to change the API URL
npm run dev                        # http://localhost:3000
```

Production build:

```bash
npm run build
npm run start
```

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `NEXT_PUBLIC_API_BASE_URL` | `http://localhost:8011` | Base URL of the RRS FastAPI service. |

Set it in `.env.local` (see `.env.local.example`). It is read in `src/lib/api.ts`, the one
place all backend fetching lives.

## Project structure

```
src/
├── app/
│   ├── layout.tsx                    # header + footer + disclaimer shell
│   ├── page.tsx                      # search (server component, reads ?q&city)
│   └── business/[id]/
│       ├── page.tsx                  # detail + p_fake histogram
│       ├── not-found.tsx             # friendly 404
│       └── reviews/page.tsx          # reviews (server-fetches first page)
├── components/
│   ├── SearchForm.tsx                # client search box → URL
│   ├── Stars.tsx                     # ★ glyph rating
│   ├── RatingCompare.tsx             # Yelp vs RRS side-by-side + CI range
│   ├── PFakeHistogram.tsx            # hand-rolled CSS bar histogram (no deps)
│   ├── ReviewCard.tsx                # one review + signal explainer
│   ├── ReviewList.tsx                # client toggle + load-more
│   └── Disclaimer.tsx                # honest-limitations copy
└── lib/
    ├── api.ts                        # typed fetch layer (the only fetch site)
    └── format.ts                     # rating/percent/date helpers + flag threshold
```

## Notes

- Charting is a dependency-free CSS bar histogram (10 bins over `p_fake` 0→1); bins at or
  above the 0.5 flag threshold are tinted.
- Server components fetch with `cache: "no-store"` so data is always live.
- 404s render a friendly page; an unreachable backend renders an inline error, never a
  crash.
