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

## Architecture

```text
raw CSVs → clean + time-split → [baseline / MF models] → offline eval (Recall@K, NDCG@K)
                                → [two-tower retrieval] → [ranking-stage reranker]
                                → FAISS index → FastAPI → Docker
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
- [x] Two-tower retrieval model
- [x] Ranking-stage model
- [x] FAISS-backed serving index
- [x] FastAPI endpoint
- [x] Dockerized deployment (Dockerfile written and builds through image-layer
      creation; final local verification blocked by this machine's disk space,
      not a code issue — see [Docker](#docker))

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
scoring at serving time would hit a never-trained embedding row.

```bash
python -m src.models.two_tower
```

| model | candidate universe | Recall@10 | NDCG@10 | Catalog coverage | Popularity bias* |
|---|---|---|---|---|---|
| matrix factorization (SVD, k=64) | 13,980 movies (≥20 ratings) | 0.0453 | 0.1859 | 10.5% | 98.4% |
| **two-tower** | **62,423 movies (full catalog)** | 0.0366 | 0.1737 | 3.39% | 94.2% |

\* share of recommended slots that are a top-decile-popularity movie — lower is more personalized, not just popular.

Not apples-to-apples with the MF row above: the two-tower model is scored
against the *entire* 62,423-movie catalog (4.5x more distractors) instead of
the ≥20-ratings 13,980-movie subset, because handling the full catalog
including movies with almost no training signal is the point of adding
content features. On that harder task it lands close to MF on accuracy and
is meaningfully less popularity-biased (94.2% vs 98.4%), but with lower raw
catalog coverage in this run — 2 epochs on CPU, not yet tuned.

The result that actually matters is the cold-start-only slice (val movies
*never seen in train* — the 29-43% of val/test movies flagged in the EDA
findings above, where baseline/MF structurally score exactly 0% by
construction since they have no embedding for an unseen ID):

| slice | users evaluated | Recall@10 | NDCG@10 |
|---|---|---|---|
| warm (all val movies) | 5,295 | 0.0366 | 0.1737 |
| **cold-start only (unseen in train)** | **3,508** | **0.0262** | **0.033** |

Nonzero recall on movies the model never trained on is the content-feature
tower doing its job — baseline and MF cannot do this at all, by construction,
regardless of how much longer they train.

## Ranking-stage model

The second stage of the two-stage pattern: a pointwise reranker
(`sklearn.ensemble.HistGradientBoostingClassifier`) that only ever reorders
the two-tower's own top-50 candidates per user — it never sees the full
catalog, so its whole job is to use signal the retrieval model's raw dot
product doesn't capture on its own: `two_tower_score`, `candidate_rank`
(the retrieval model's own ordering), `log_item_popularity`, `genre_overlap`
(between a user's aggregated train-history genre vector and the candidate's
genres), `has_genome`, `log_user_activity` (`src/ranking/features.py`).

Trained on 80% of val users (4,236 users, 211,800 candidate rows, 11.86%
positive rate), scored on the other 20% it never trained on, then checked a
second time against `test.parquet` — the first and only place the test split
is touched anywhere in this project — for a genuinely held-out number, not
one measured on data the model was fit to:

```bash
python -m src.ranking.train
```

| slice | n users | | Recall@10 | NDCG@10 | Catalog coverage | Popularity bias* |
|---|---|---|---|---|---|---|
| holdout (val, unseen by ranker) | 1,059 | before rerank | 0.0403 | 0.1828 | 1.89% | 94.2% |
| holdout (val, unseen by ranker) | 1,059 | **after rerank** | **0.0442** | **0.2081** | 1.64% | **86.6%** |
| test set | 4,099 | before rerank | 0.0263 | 0.1158 | 3.28% | 94.6% |
| test set | 4,099 | **after rerank** | **0.0272** | **0.1185** | 2.81% | **87.6%** |

\* share of recommended slots that are a top-decile-popularity movie — lower is more personalized, not just popular.

**What worked**: reranking lifts both accuracy metrics on both slices —
holdout Recall@10 +9.7%, NDCG@10 +13.8%; test set +3.4% / +2.3% (smaller,
as expected on data the ranker never touched during training, but still a
genuine, consistent-direction lift, not noise). Popularity bias drops
substantially on both slices (94%→86.6% holdout, 94.6%→87.6% test) — the
item-popularity and genre-overlap features actively pull the ranking away
from "whatever's popular" and toward what actually matches the user's
history, which the raw two-tower dot product alone doesn't do as well.
**What didn't**: catalog
coverage drops slightly on both slices (e.g. 1.89%→1.64% on holdout) — a
pointwise classifier optimizing per-item relevance has no term rewarding
spreading recommendations across the catalog, so accuracy and coverage pull
in different directions here, same tension already visible in the
baseline/MF comparison above. Pairwise/listwise reranking (e.g. LambdaMART)
would likely rank better than this pointwise setup but pulls in a
LightGBM/XGBoost dependency this project deliberately avoided — noted as
future work, the same treatment as the IVF `nprobe` tuning gap below.

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

## Serving: FAISS index + FastAPI

`src/serving/build_index.py` builds everything the API needs to answer a
request without touching raw training data: a FAISS `IndexFlatIP` over the
full-catalog item embeddings (exact, not approximate — the latency
benchmark above already found exact search is sub-millisecond at this
catalog's real scale, so there's no reason to trade accuracy for a latency
win nobody needs), plus the ranking-stage feature arrays and a compact
boolean seen-items mask.

```bash
python -m src.serving.build_index   # writes Data/processed/serving/
uvicorn api.main:app --reload       # GET /health, GET /recommend/{user_id}?k=10
```

**What worked**: keeping the servable artifact set small mattered more than
it first looked — the tag-genome content features are 59MB raw, but the
ranker only ever uses a 1-bit `has_genome` flag derived from them, so that
59MB matrix is loaded once at index-build time and never again; the API
loads a ~90MB total footprint (checkpoint + FAISS index + feature arrays +
ranker) instead. The API also never reconstructs the item tower at all —
item embeddings are precomputed into the FAISS index ahead of time, so
serving only needs the small user tower to embed the incoming user ID.

Example request/response:

```bash
$ curl http://localhost:8000/recommend/1?k=5
{
  "user_id": 1,
  "personalized": true,
  "recommendations": [
    {"movie_id": 4226, "title": "Memento (2000)", "score": 0.171},
    {"movie_id": 4878, "title": "Donnie Darko (2001)", "score": 0.165},
    {"movie_id": 5618, "title": "Spirited Away (Sen to Chihiro no kamikakushi) (2001)", "score": 0.165},
    {"movie_id": 2959, "title": "Fight Club (1999)", "score": 0.159},
    {"movie_id": 6874, "title": "Kill Bill: Vol. 1 (2003)", "score": 0.157}
  ]
}
```

An unknown `user_id` (never seen in train) falls back to the trained UNK
embedding row instead of erroring, and the response flags `"personalized": false`
so a caller can tell the difference — the same cold-start path
the two-tower model was explicitly trained to support, tested live:

```bash
$ curl http://localhost:8000/recommend/999999999?k=3
{"user_id": 999999999, "personalized": false, "recommendations": [
  {"movie_id": 593, "title": "Silence of the Lambs, The (1991)", "score": 3.91},
  {"movie_id": 318, "title": "Shawshank Redemption, The (1994)", "score": 3.91},
  {"movie_id": 356, "title": "Forrest Gump (1994)", "score": 3.90}
]}
```

If `Data/processed/serving/` hasn't been built yet, `/health` reports
`"status": "degraded"` and `/recommend` returns `503` instead of crashing —
this is what keeps `api/main.py` importable and testable in CI, which has
no `Data/` at all (see `tests/test_api.py`).

## Docker

```bash
docker build -t movie-recommender .
docker run -p 8000:8000 movie-recommender
```

Serving-only image — it doesn't train anything at build time (two-tower
training alone takes ~27 min on CPU); it assumes the artifacts above already
exist locally and copies them in. `torch`'s default PyPI resolution pulls
the multi-gigabyte CUDA build, so the `Dockerfile` installs the CPU-only
wheel explicitly first. `.dockerignore` excludes the raw `Data/ml-25m/` CSVs
and the training-only parquet/genome files by exact path, so the image only
ships the ~90MB of serving artifacts, not the full processed dataset.

The image builds cleanly through dependency install and layer creation; the
final local verification (`docker run` + hitting `/health` from inside the
container) is blocked on this development machine by disk space (under 1GB
free), not by anything in the Dockerfile — re-run `docker build` once space
is available to confirm end-to-end.

## Testing & CI

41 unit tests (`tests/`) cover the parts of the pipeline that are easy to
get subtly wrong and hard to notice from the metrics alone: dedup/cleaning
edge cases, that the time-based split doesn't leak future rows into train,
that cold-start detection is counted correctly, the ranking metrics against
hand-computed examples, checkpoint save/load round-tripping, ranking-feature
engineering, and the FastAPI routes (via dependency injection against small
in-memory fakes, so no test needs the real trained model on disk). Runs on
every push to `main` via GitHub Actions (`.github/workflows/ci.yml`) — none
of it depends on `Data/` existing or the two-tower model being trained,
since neither is present in a fresh CI checkout.

```bash
python -m pytest tests/ -v
```

## Repo layout

```text
src/data/       loading, cleaning, time-split, EDA
src/models/     item-item CF, matrix factorization, and two-tower retrieval models
src/ranking/    ranking-stage feature engineering + training (rerank over two-tower candidates)
src/serving/    builds the FAISS index + serving artifacts the API loads
src/eval/       ranking metrics (Recall@K, NDCG@K) + diversity metrics (coverage, popularity bias)
api/            FastAPI serving layer (schemas, recommender service, routes)
tests/          unit tests for src/data, src/eval, src/ranking, and api/
.github/        CI workflow
notebooks/      exploratory notebooks only — no pipeline logic lives here
docs/           design notes, EDA figures/summary, model + ranker eval, latency results
Data/           raw + processed data (gitignored)
Dockerfile      serving-only image (see Docker section)
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m src.data.build_dataset   # builds Data/processed/{train,val,test}.parquet
python -m src.data.eda             # writes docs/eda_summary.json + figures
python -m src.models.compare       # trains + evaluates baseline/MF, benchmarks FAISS latency
python -m src.models.two_tower     # trains two-tower retrieval model (~27 min CPU)
python -m src.ranking.train        # trains ranker, writes ranker_model.joblib
python -m src.serving.build_index  # builds FAISS index + serving artifacts
python -m pytest tests/ -v         # runs the unit test suite
uvicorn api.main:app --reload      # run the API locally
docker build -t movie-recommender .
docker run -p 8000:8000 movie-recommender
```
