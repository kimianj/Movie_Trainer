# Movie Recommendation System

A production-style movie recommendation system built end-to-end on MovieLens-25M:
raw interaction data → cleaned/split dataset → candidate generation + ranking
models → evaluation → a served API. Structured the way a recsys team would
structure it, not as a single notebook.

## Problem

Given a user's rating history, predict which movies they're most likely to
engage with next — and do it fast enough to serve at request time, not just
score well offline.

## Architecture (planned)

```text
raw CSVs → clean + time-split → [baseline model] → offline eval (Recall@K, NDCG@K)
                                → [two-tower model] → embeddings → FAISS index → FastAPI
```

Two-stage pattern: cheap, broad **candidate generation** (two-tower retrieval
over the full catalog via approximate nearest-neighbor search) followed by a
narrower, more precise **ranking** stage. This mirrors how YouTube/Meta/Amazon
scale recommendation past the point where scoring every item per request is
feasible.

## Status

- [x] Data layer: load, clean, time-based split, EDA
- [x] Baseline model (item-item collaborative filtering)
- [ ] Two-tower retrieval model
- [ ] Ranking-stage model
- [ ] Offline evaluation (Recall@K, NDCG@K)
- [ ] FAISS-backed serving index
- [ ] FastAPI endpoint
- [ ] Dockerized deployment

## Data

[MovieLens-25M](https://grouplens.org/datasets/movielens/25m/): 25,000,095
ratings from 162,541 users on 62,423 movies, 1995-01-09 to 2019-11-21. Users
are pre-filtered by GroupLens to have rated at least 20 movies; no further
filtering was needed — the raw data has zero exact duplicates, invalid
rating values, invalid timestamps, or orphan movie IDs (see
`docs/eda_summary.json` and `Data/processed/split_report.json` for the full
numbers). Not checked into git (`Data/` is gitignored); download and unzip
into `Data/ml-25m/` to reproduce.

### Split strategy

A **global time-based split** (80% train / 10% val / 10% test by row count,
cut on timestamp across all users at once) — not random, not per-user. This
matches production: the system is trained on the past and evaluated on the
future. Reproducible via:

```bash
python -m src.data.build_dataset
```

| split | date range              | rows       | users   | movies |
|-------|--------------------------|------------|---------|--------|
| train | 1995-01-09 – 2016-06-25 | 20,000,076 | 137,883 | 34,461 |
| val   | 2016-06-25 – 2018-01-04 | 2,500,009  | 18,003  | 37,221 |
| test  | 2018-01-04 – 2019-11-21 | 2,500,010  | 18,442  | 49,405 |

### EDA findings that shape the modeling plan

Run via `python -m src.data.eda` (writes `docs/eda_summary.json` and
`docs/figures/`).

- **Extremely sparse**: 0.26% matrix density (25M ratings / 162K users × 59K
  movies) — standard for recsys, but rules out dense matrix methods at scale.
- **Popularity long tail**: the top 10% of movies account for 94% of all
  ratings (`docs/figures/movie_popularity_long_tail.png`). Random negative
  sampling during training will oversample obscure titles relative to what a
  user would realistically be choosing between; needs popularity-aware
  negative sampling.
- **Heavy cold-start pressure in val/test, and it's real, not a split
  artifact**: 70-78% of val/test *users* and 29-43% of val/test *movies* never
  appear in train, touching 85-90% of val/test rows. Checked whether this was
  an artifact of the split (e.g. under-sampling) — it isn't: **100% of
  cold-start movies in val/test were first ever rated after the train
  cutoff**, i.e. they entered the MovieLens catalog later (`docs/figures/ratings_per_day.png`
  shows the full ratings-per-day timeline with the split cuts marked). This is
  the real challenge a production system faces (new users, new releases,
  every day) and it directly motivates using **content features (genres,
  tag-genome scores) in the item tower**, not a pure ID-embedding lookup —
  a model that only memorizes IDs it saw in training cannot say anything about
  a movie added after training ended.
- **Rating distribution**: mean 3.53, std 1.06, skewed toward positive ratings
  (`docs/figures/rating_distribution.png`) — worth keeping in mind when
  turning ratings into implicit positive/negative labels for the two-tower
  model.

## Baseline: item-item collaborative filtering

Cosine similarity over the binary user-item interaction matrix ("people who
watched X also watched Y") — no learned parameters, no training loop. Its job
is to prove the split → model → eval pipeline works end-to-end and to set a
number the two-tower model has to beat. Run via:

```bash
python -m src.models.baseline
```

| | |
|---|---|
| Candidate universe | 13,980 movies (≥20 train ratings; 20,481 long-tail movies excluded) |
| Val users evaluated | 5,217 / 18,003 (29.4% — the model has no history for the rest, see cold-start finding above) |
| Recall@10 | 0.0322 |
| NDCG@10 | 0.1523 |

Both coverage numbers are limitations of *this* model, not artifacts to
paper over: it can only recommend from the pre-filtered candidate universe,
and it can only be evaluated on users it saw in train. The two-tower model's
job is to close both gaps — full catalog coverage via content features, and
warm-start-independent scoring for new users via richer signals.

Qualitative sanity check (nearest neighbors to *Toy Story (1995)*): Star Wars
IV, Independence Day, Forrest Gump, Back to the Future, Star Wars VI,
Jurassic Park, Mission: Impossible, The Matrix, Toy Story 2, Star Wars V —
all mainstream 90s titles plus its own sequel, which is the expected shape
for a co-occurrence-based similarity.

## Repo layout

```text
src/data/       loading, cleaning, time-split, EDA
src/models/     item-item CF baseline (done); two-tower model (WIP)
src/eval/       ranking metrics (Recall@K, NDCG@K)
api/            FastAPI serving layer (WIP)
notebooks/      exploratory notebooks only — no pipeline logic lives here
docs/           design notes, EDA figures/summary
Data/           raw + processed data (gitignored)
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m src.data.build_dataset   # builds Data/processed/{train,val,test}.parquet
python -m src.data.eda             # writes docs/eda_summary.json + figures
python -m src.models.baseline      # trains + evaluates the item-item CF baseline
```
