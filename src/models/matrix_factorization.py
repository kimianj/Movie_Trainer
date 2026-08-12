"""Matrix factorization baseline via truncated SVD over the implicit
user-item interaction matrix.

Sits between the item-item CF baseline and the (future) two-tower model:
still a shallow, classical latent-factor method, but it learns dense
user/item embeddings instead of relying on raw co-occurrence, which is what
makes an embedding-based ANN search (FAISS) meaningful to benchmark against
brute force -- the same serving pattern the two-tower model will use later.

Usage:
    python -m src.models.matrix_factorization
"""

import json

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

from src.data.config import PROCESSED_DIR, RANDOM_SEED, ROOT_DIR
from src.eval.metrics import evaluate

# Same floor as the item-item baseline, so both models share one candidate
# universe and the comparison is apples-to-apples rather than one model
# just having an easier catalog to work with.
MIN_ITEM_TRAIN_RATINGS = 20
N_FACTORS = 64
TOP_K = 10
RECOMMEND_BATCH_SIZE = 1000


class MatrixFactorization:
    def __init__(self, n_factors: int = N_FACTORS, min_item_train_ratings: int = MIN_ITEM_TRAIN_RATINGS):
        self.n_factors = n_factors
        self.min_item_train_ratings = min_item_train_ratings

    def fit(self, train: pd.DataFrame) -> "MatrixFactorization":
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
        data = np.ones(len(train), dtype=np.float32)
        self.X_ = sparse.csr_matrix(
            (data, (rows, cols)), shape=(len(self.user_ids_), len(self.item_ids_))
        )

        svd = TruncatedSVD(n_components=self.n_factors, random_state=RANDOM_SEED)
        self.user_factors_ = svd.fit_transform(self.X_).astype(np.float32)
        self.item_factors_ = svd.components_.T.astype(np.float32)
        return self

    def recommend_batch(self, user_ids: list, k: int) -> dict:
        known = [u for u in user_ids if u in self.user_to_idx_]
        recs = {}
        for start in range(0, len(known), RECOMMEND_BATCH_SIZE):
            batch = known[start : start + RECOMMEND_BATCH_SIZE]
            idx = [self.user_to_idx_[u] for u in batch]
            seen = self.X_[idx]
            scores = self.user_factors_[idx] @ self.item_factors_.T
            scores[seen.toarray() > 0] = -np.inf
            top_k_idx = np.argpartition(-scores, k, axis=1)[:, :k]
            for row, user_id in enumerate(batch):
                row_scores = scores[row, top_k_idx[row]]
                order = np.argsort(-row_scores)
                ranked_idx = top_k_idx[row][order]
                recs[user_id] = [
                    self.item_ids_[i] for i in ranked_idx if scores[row, i] > -np.inf
                ]
        return recs


def main() -> None:
    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    val = pd.read_parquet(PROCESSED_DIR / "val.parquet")

    print(f"Fitting matrix factorization (k={N_FACTORS} factors)...")
    model = MatrixFactorization().fit(train)
    print(
        f"  candidate universe: {len(model.item_ids_):,} movies "
        f"({model.dropped_movie_count_:,} long-tail movies excluded)"
    )

    val_users = set(val["userId"].unique())
    warm_val_users = val_users & set(model.user_ids_)
    item_universe = set(model.item_ids_)
    relevant = {}
    for user_id, group in val[val["userId"].isin(warm_val_users)].groupby("userId"):
        relevant[user_id] = set(group["movieId"]) & item_universe

    print(f"Generating top-{TOP_K} recommendations for {len(relevant):,} warm val users...")
    recs = model.recommend_batch(list(relevant.keys()), TOP_K)

    metrics = evaluate(recs, relevant, TOP_K)
    result = {
        "metrics": metrics,
        "coverage": {
            "val_users_total": len(val_users),
            "val_users_warm_pct": round(100 * len(warm_val_users) / len(val_users), 1),
            "candidate_movies": len(model.item_ids_),
            "candidate_movies_excluded_long_tail": model.dropped_movie_count_,
        },
    }
    print(json.dumps(result, indent=2))

    with open(ROOT_DIR / "docs" / "mf_eval.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
