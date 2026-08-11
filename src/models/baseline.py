"""Item-item collaborative filtering baseline.

Recommends movies similar to what a user has already watched, where
"similar" means "rated by the same users" (cosine similarity over the
binary user-item interaction matrix). No training loop, no learned
parameters beyond co-occurrence counts -- this exists to prove the
data -> model -> eval pipeline works end-to-end and to set a benchmark
number the two-tower model has to beat.

Usage:
    python -m src.models.baseline
"""

import json

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

from src.data.config import PROCESSED_DIR, ROOT_DIR
from src.eval.metrics import evaluate

# Movies with fewer training ratings than this are excluded from the
# candidate universe. Item-item similarity for a movie rated by 2 people is
# noise, and without this cap the item x item similarity matrix (up to
# 34,461 x 34,461 here) doesn't fit comfortably in memory on a CPU-only
# machine. This is a real capacity/quality trade-off, not just a speed hack:
# it means this baseline structurally cannot recommend long-tail movies,
# which is exactly the gap the two-tower model's content features are meant
# to close later.
MIN_ITEM_TRAIN_RATINGS = 20
TOP_K = 10
RECOMMEND_BATCH_SIZE = 1000


class ItemItemCF:
    def __init__(self, min_item_train_ratings: int = MIN_ITEM_TRAIN_RATINGS):
        self.min_item_train_ratings = min_item_train_ratings

    def fit(self, train: pd.DataFrame) -> "ItemItemCF":
        item_counts = train.groupby("movieId").size()
        kept_movies = item_counts[item_counts >= self.min_item_train_ratings].index
        self.dropped_movie_count_ = train["movieId"].nunique() - len(kept_movies)

        train = train[train["movieId"].isin(kept_movies)]

        self.user_ids_ = np.sort(train["userId"].unique())
        self.item_ids_ = np.sort(train["movieId"].unique())
        self.user_to_idx_ = {u: i for i, u in enumerate(self.user_ids_)}
        self.item_to_idx_ = {m: i for i, m in enumerate(self.item_ids_)}

        rows = train["userId"].map(self.user_to_idx_).to_numpy()
        cols = train["movieId"].map(self.item_to_idx_).to_numpy()
        # Implicit feedback: any rating counts as a positive interaction, regardless
        # of star value, since the task is "what will they engage with next", not
        # "what rating will they give it".
        data = np.ones(len(train), dtype=np.float32)
        self.X_ = sparse.csr_matrix(
            (data, (rows, cols)), shape=(len(self.user_ids_), len(self.item_ids_))
        )

        item_sim = cosine_similarity(self.X_.T, dense_output=False).tocsr()
        item_sim.setdiag(0)
        item_sim.eliminate_zeros()
        self.item_sim_ = item_sim
        return self

    def recommend_batch(self, user_ids: list, k: int) -> dict:
        """Top-k recommendations for users already seen during fit. Unknown
        (cold-start) user_ids are silently skipped -- this model has no
        history to work from for them, which is the point being measured.
        """
        known = [u for u in user_ids if u in self.user_to_idx_]
        recs = {}
        for start in range(0, len(known), RECOMMEND_BATCH_SIZE):
            batch = known[start : start + RECOMMEND_BATCH_SIZE]
            idx = [self.user_to_idx_[u] for u in batch]
            seen = self.X_[idx]
            scores = (seen @ self.item_sim_).toarray()
            scores[seen.toarray() > 0] = -np.inf  # exclude already-seen items
            top_k_idx = np.argpartition(-scores, k, axis=1)[:, :k]
            for row, user_id in enumerate(batch):
                row_scores = scores[row, top_k_idx[row]]
                order = np.argsort(-row_scores)
                ranked_idx = top_k_idx[row][order]
                recs[user_id] = [
                    self.item_ids_[i] for i in ranked_idx if scores[row, i] > -np.inf
                ]
        return recs

    def similar_movies(self, movie_id: int, k: int = 10) -> list:
        if movie_id not in self.item_to_idx_:
            return []
        idx = self.item_to_idx_[movie_id]
        sims = self.item_sim_[idx].toarray().ravel()
        top_idx = np.argsort(-sims)[:k]
        return [(self.item_ids_[i], float(sims[i])) for i in top_idx if sims[i] > 0]


def main() -> None:
    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    val = pd.read_parquet(PROCESSED_DIR / "val.parquet")
    movies = pd.read_parquet(PROCESSED_DIR / "movies.parquet")

    print(f"Fitting item-item CF (min {MIN_ITEM_TRAIN_RATINGS} train ratings/movie)...")
    model = ItemItemCF(min_item_train_ratings=MIN_ITEM_TRAIN_RATINGS).fit(train)
    print(
        f"  candidate universe: {len(model.item_ids_):,} movies "
        f"({model.dropped_movie_count_:,} long-tail movies excluded)"
    )

    # Coverage: this model can only be evaluated on val users who also appear
    # in train, and only against val items inside the candidate universe.
    # Both numbers are reported, not hidden, because they materially limit
    # what "Recall@10" means here.
    val_users = set(val["userId"].unique())
    warm_val_users = val_users & set(model.user_ids_)
    item_universe = set(model.item_ids_)

    relevant = {}
    for user_id, group in val[val["userId"].isin(warm_val_users)].groupby("userId"):
        relevant[user_id] = set(group["movieId"]) & item_universe

    print(
        f"  val user coverage: {len(warm_val_users):,} / {len(val_users):,} "
        f"({100 * len(warm_val_users) / len(val_users):.1f}%) users seen in train"
    )

    print(f"Generating top-{TOP_K} recommendations for {len(relevant):,} warm val users...")
    recs = model.recommend_batch(list(relevant.keys()), TOP_K)

    metrics = evaluate(recs, relevant, TOP_K)
    coverage = {
        "val_users_total": len(val_users),
        "val_users_warm_pct": round(100 * len(warm_val_users) / len(val_users), 1),
        "candidate_movies": len(model.item_ids_),
        "candidate_movies_excluded_long_tail": model.dropped_movie_count_,
    }
    result = {"metrics": metrics, "coverage": coverage}
    print(json.dumps(result, indent=2))

    with open(ROOT_DIR / "docs" / "baseline_eval.json", "w") as f:
        json.dump(result, f, indent=2)

    # Qualitative sanity check: do the nearest neighbors look sane?
    sample_title = "Toy Story (1995)"
    sample_id = movies.loc[movies["title"] == sample_title, "movieId"]
    if len(sample_id):
        print(f"\nMovies most similar to '{sample_title}':")
        for movie_id, sim in model.similar_movies(int(sample_id.iloc[0]), k=10):
            title = movies.loc[movies["movieId"] == movie_id, "title"].iloc[0]
            print(f"  {sim:.3f}  {title}")


if __name__ == "__main__":
    main()
