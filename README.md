# Movie Recommendation System

![CI](https://github.com/kimianj/Movie_Trainer/actions/workflows/ci.yml/badge.svg)

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
- [x] Matrix factorization model (SVD) + model comparison
- [x] Offline evaluation: Recall@K, NDCG@K, catalog coverage, popularity bias
- [x] Retrieval latency benchmark: brute-force vs FAISS (exact + approximate)
- [x] Unit tests + CI
- [x] Two-tower retrieval model (implemented; training/eval in progress)
- [ ] Ranking-stage model
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

## Models

Both models below share one candidate universe (13,980 movies with ≥20 train
ratings; 20,481 long-tail movies excluded — similarity/factors learned from
a handful of ratings are noise) and are evaluated on the same 5,217 warm val
users (29.4% of val users — the rest have no train history, see the
cold-start finding above), so the comparison is apples-to-apples.

**Item-item collaborative filtering** (`src/models/baseline.py`) — cosine
similarity over the binary user-item interaction matrix ("people who watched
X also watched Y"). No learned parameters, no training loop; exists to prove
the split → model → eval pipeline works end-to-end.

**Matrix factorization** (`src/models/matrix_factorization.py`) — truncated
SVD (k=64) over the same matrix, producing dense user/item embeddings. Still
a shallow, classical method, but embedding-based rather than co-occurrence-
based — the same serving pattern (embeddings → nearest-neighbor search) the
two-tower model will use later.

```bash
python -m src.models.baseline
python -m src.models.matrix_factorization
python -m src.models.compare   # runs both + a popularity baseline side by side
```

| model | Recall@10 | NDCG@10 | Catalog coverage | Popularity bias* |
|---|---|---|---|---|
| popularity (always top-10 most-rated) | 0.0131 | 0.0495 | 0.1% | 100% |
| item-item CF | 0.0322 | 0.1523 | 4.7% | 99.7% |
| **matrix factorization (SVD, k=64)** | **0.0453** | **0.1859** | **10.5%** | 98.4% |

\* share of recommended slots that are a top-decile-popularity movie — lower is more personalized, not just popular.

**What worked**: matrix factorization beats item-item CF on every accuracy
metric and covers 2x more of the catalog. **What didn't**: all three models,
including the two "personalized" ones, fill 98%+ of their recommendation
slots with top-decile-popular movies — on this sparse implicit-feedback data,
plain collaborative filtering barely escapes recommending whatever's already
popular. This is the concrete reason the two-tower model needs
popularity-aware negative sampling (flagged in the EDA section above), not
just a bigger model — accuracy alone doesn't fix this, the training
objective has to.

Qualitative sanity check (item-item CF nearest neighbors to *Toy Story
(1995)*): Star Wars IV, Independence Day, Forrest Gump, Back to the Future,
Star Wars VI, Jurassic Park, Mission: Impossible, The Matrix, Toy Story 2,
Star Wars V — all mainstream 90s titles plus its own sequel, the expected
shape for a co-occurrence-based similarity.

**Two-tower retrieval model** (`src/models/two_tower.py`) — a user tower
(ID embedding only; MovieLens has no user side-info) and an item tower (ID
embedding + genre + tag-genome content features), trained with in-batch
sampled softmax and a log-Q popularity correction so the model isn't just
rewarded for recommending whatever's already popular — the concrete failure
mode the baseline/MF comparison above found. Directly targets the two
findings from EDA: index 0 is reserved as UNK in both towers, and each
example's ID is randomly zeroed out to UNK during training (`ID_DROPOUT_PROB
= 0.15`) so the UNK row actually receives gradient and the item tower is
forced to sometimes rely on content features alone — without this, cold-start
scoring at serving time would hit a never-trained embedding row. Evaluated
on the full catalog (not just the ≥20-ratings subset the baseline/MF use)
plus a cold-start-only slice (val movies never seen in train, where
baseline/MF structurally score 0% by construction) to directly test whether
the content features work.

```bash
python -m src.models.two_tower
```

Training/evaluation is in progress — results (`docs/two_tower_eval.json`)
to follow once the run completes.

## Retrieval latency: brute-force vs FAISS

The pitch for two-stage retrieval only holds if approximate nearest-neighbor
search is actually faster where it matters — so this benchmarks brute-force
numpy, FAISS exact search (`IndexFlatIP`), and FAISS approximate search
(`IndexIVFFlat`, nprobe=8) over the matrix factorization's 64-dim item
embeddings, both at the real catalog size and at synthetic sizes matched to
a real streaming catalog (`src/models/compare.py`, since 14K movies is too
small to ever be a bottleneck).

| catalog size | brute-force | FAISS exact | FAISS approx (IVF) | IVF overlap w/ exact@10 |
|---|---|---|---|---|
| 13,980 (real) | 0.23 ms | 0.015 ms | 0.004 ms | 72% |
| 100,000 (synthetic) | 0.72 ms | 0.22 ms | 0.018 ms | 23% |
| 1,000,000 (synthetic) | 13.9 ms | 3.6 ms | 0.058 ms | 17% |

**What worked**: at 1M items, FAISS IVF is ~240x faster than brute force
(0.058ms vs 13.9ms per query) — this is where two-stage retrieval actually
earns its keep. **What didn't**: at the real 14K-movie catalog, brute force
is already sub-millisecond, so approximate search buys latency nobody needs
yet. And the IVF overlap-with-exact column is the trade-off usually left out
of "FAISS is faster" claims — at default settings (nlist=√N, nprobe=8), the
approximate index only agrees with the exact top-10 17-23% of the time at
scale. That's too lossy to ship as-is; a real deployment would need to tune
nprobe upward (fewer, more thorough cluster probes) or switch to HNSW, and
re-measure this same latency/overlap curve before trusting it.

## Testing & CI

22 unit tests (`tests/`) cover the parts of the pipeline that are easy to
get subtly wrong and hard to notice from the metrics alone: dedup/cleaning
edge cases, that the time-based split doesn't leak future rows into train,
that cold-start detection is counted correctly, and the ranking metrics
against hand-computed examples. Runs on every push to `main` via GitHub
Actions (`.github/workflows/ci.yml`).

```bash
python -m pytest tests/ -v
```

## Repo layout

```text
src/data/       loading, cleaning, time-split, EDA
src/models/     item-item CF + matrix factorization baselines (done); two-tower model (WIP)
src/eval/       ranking metrics (Recall@K, NDCG@K) + diversity metrics (coverage, popularity bias)
api/            FastAPI serving layer (WIP)
tests/          unit tests for src/data and src/eval
.github/        CI workflow
notebooks/      exploratory notebooks only — no pipeline logic lives here
docs/           design notes, EDA figures/summary, model comparison + latency results
Data/           raw + processed data (gitignored)
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m src.data.build_dataset   # builds Data/processed/{train,val,test}.parquet
python -m src.data.eda             # writes docs/eda_summary.json + figures
python -m src.models.compare       # trains + evaluates all models, benchmarks FAISS latency
python -m pytest tests/ -v         # runs the unit test suite
```
