"""Feature engineering for the ranking stage, shared between offline
training (src/ranking/train.py) and online serving (api/recommender.py) so
there's exactly one implementation of each feature -- computing them twice
would risk train/serve skew.

The ranking model only ever reorders candidates the two-tower retrieval
model already surfaced (CANDIDATE_N per user), so every feature here is
either the retrieval model's own signal (two_tower_score, candidate_rank)
or a cheap, precomputable signal about the user/item pair (popularity,
genre overlap, activity, content availability).
"""

import numpy as np
import pandas as pd

CANDIDATE_N = 50

FEATURE_COLUMNS = [
    "two_tower_score",
    "candidate_rank",
    "log_item_popularity",
    "genre_overlap",
    "has_genome",
    "log_user_activity",
]


def build_user_genre_profile(
    train: pd.DataFrame, user_to_idx: dict, catalog_movie_to_idx: dict, genre_matrix: np.ndarray
) -> np.ndarray:
    """Mean genre vector over each user's train-period movies. Row i
    corresponds to user_to_idx[u] - 1, the same 0-indexed convention
    TwoTowerRecommender.X_ uses.
    """
    n_users = len(user_to_idx)
    genre_dim = genre_matrix.shape[1]
    user_idx = train["userId"].map(user_to_idx).to_numpy(dtype=np.int64) - 1
    catalog_idx = train["movieId"].map(catalog_movie_to_idx).to_numpy(dtype=np.int64)

    sums = np.zeros((n_users, genre_dim), dtype=np.float64)
    counts = np.zeros(n_users, dtype=np.float64)
    np.add.at(sums, user_idx, genre_matrix[catalog_idx])
    np.add.at(counts, user_idx, 1.0)
    counts[counts == 0] = 1.0  # no train rows for this user shouldn't happen, but avoid div/0
    return (sums / counts[:, None]).astype(np.float32)


def build_item_popularity(train: pd.DataFrame, catalog_movie_ids: np.ndarray) -> np.ndarray:
    """log1p(train rating count) per catalog item, aligned to catalog_movie_ids
    order; 0 for catalog items absent from train (cold items)."""
    counts = train["movieId"].value_counts().reindex(catalog_movie_ids, fill_value=0)
    return np.log1p(counts.to_numpy(dtype=np.float64)).astype(np.float32)


def build_user_activity(train: pd.DataFrame, user_ids: np.ndarray) -> np.ndarray:
    """log1p(train rating count) per user, aligned to user_ids order (the
    same order build_vocab assigns user_to_idx from, so user_to_idx[u] - 1
    indexes into this array correctly)."""
    counts = train["userId"].value_counts().reindex(user_ids, fill_value=0)
    return np.log1p(counts.to_numpy(dtype=np.float64)).astype(np.float32)


def build_candidate_features(
    candidates: dict,
    user_genre_profile: np.ndarray,
    user_activity: np.ndarray,
    item_popularity: np.ndarray,
    genre_matrix: np.ndarray,
    has_genome: np.ndarray,
    user_to_idx: dict,
    catalog_movie_to_idx: dict,
) -> pd.DataFrame:
    """candidates: user_id -> ranked [(movieId, two_tower_score), ...], e.g.
    from TwoTowerRecommender.recommend_batch_with_scores(users, CANDIDATE_N).
    Returns one row per (user_id, movie_id) with FEATURE_COLUMNS plus
    user_id/movie_id identifier columns.
    """
    rows = []
    for user_id, ranked in candidates.items():
        user_row = user_to_idx[user_id] - 1
        for rank, (movie_id, score) in enumerate(ranked, start=1):
            catalog_idx = catalog_movie_to_idx[movie_id]
            rows.append(
                {
                    "user_id": user_id,
                    "movie_id": movie_id,
                    "two_tower_score": float(score),
                    "candidate_rank": rank,
                    "log_item_popularity": float(item_popularity[catalog_idx]),
                    "genre_overlap": float(
                        np.dot(user_genre_profile[user_row], genre_matrix[catalog_idx])
                    ),
                    "has_genome": float(has_genome[catalog_idx]),
                    "log_user_activity": float(user_activity[user_row]),
                }
            )
    return pd.DataFrame(rows)


def label_candidates(feature_df: pd.DataFrame, relevant: dict) -> pd.Series:
    """1 if (user_id, movie_id) is in that user's relevant set, else 0."""
    positive_pairs = {(u, m) for u, items in relevant.items() for m in items}
    keys = list(zip(feature_df["user_id"], feature_df["movie_id"]))
    return pd.Series([1 if key in positive_pairs else 0 for key in keys], index=feature_df.index)


def rerank(model, feature_df: pd.DataFrame, k: int = 10) -> dict:
    """Score feature_df with model.predict_proba, sort each user's
    candidates by predicted probability of relevance, truncate to k.
    Returns {user_id: [movie_id, ...]} -- the same shape evaluate() expects.
    """
    scores = model.predict_proba(feature_df[FEATURE_COLUMNS])[:, 1]
    scored = feature_df.assign(predicted_score=scores)
    result = {}
    for user_id, group in scored.groupby("user_id"):
        top = group.sort_values("predicted_score", ascending=False).head(k)
        result[user_id] = top["movie_id"].tolist()
    return result
