"""Ranking-stage model: a pointwise reranker over the two-tower retrieval
model's own top-CANDIDATE_N candidates per user. This is the second stage
of the two-stage pattern the README documents (cheap, broad candidate
generation -> narrower, more precise ranking) -- the ranker never sees the
full catalog, only what retrieval already surfaced, so its only job is to
reorder those candidates using signal the retrieval model's raw dot product
doesn't capture on its own (item popularity, how well an item's genres match
the user's history, content availability, user activity level).

Trained on a held-out 20% slice of val users the ranker itself never trains
on (not the same val users used to pick anything upstream), with a second,
single-use check against test.parquet -- the first and only place the test
split is touched anywhere in this project -- so the reported lift is a
genuine held-out number, not one measured on the data it was fit to.

Usage:
    python -m src.ranking.train
"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from src.data.config import PROCESSED_DIR, RANDOM_SEED, ROOT_DIR
from src.eval.diversity import catalog_coverage, popularity_bias, top_decile_items
from src.eval.metrics import evaluate
from src.models.two_tower import TwoTowerRecommender, load_catalog_features, load_checkpoint
from src.ranking.features import (
    CANDIDATE_N,
    FEATURE_COLUMNS,
    build_candidate_features,
    build_item_popularity,
    build_user_activity,
    build_user_genre_profile,
    label_candidates,
    rerank,
)

TOP_K = 10
HOLDOUT_FRACTION = 0.2
RANKER_CHECKPOINT_PATH = PROCESSED_DIR / "ranker_model.joblib"


def build_relevant(df: pd.DataFrame, warm_users: np.ndarray, item_universe: set) -> dict:
    relevant = {}
    subset = df[df["userId"].isin(warm_users)]
    for user_id, group in subset.groupby("userId"):
        relevant[user_id] = set(group["movieId"]) & item_universe
    return relevant


def eval_slice(recs: dict, relevant: dict, item_universe: set, decile_items: set) -> dict:
    metrics = evaluate(recs, relevant, TOP_K)
    return {
        **metrics,
        "catalog_coverage": catalog_coverage(recs, item_universe),
        "popularity_bias_top_decile_share": popularity_bias(recs, decile_items),
    }


def main() -> None:
    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    val = pd.read_parquet(PROCESSED_DIR / "val.parquet")
    test = pd.read_parquet(PROCESSED_DIR / "test.parquet")

    print("Loading checkpoint + catalog features...")
    features = load_catalog_features()
    user_tower, item_tower, user_to_idx, item_to_idx, features = load_checkpoint(features=features)
    model = TwoTowerRecommender(user_tower, item_tower, user_to_idx, item_to_idx, features, train)

    genre_matrix = features["genre_matrix"].numpy()
    has_genome = features["has_genome"].numpy()
    catalog_movie_to_idx = features["catalog_movie_to_idx"]
    item_universe = set(model.item_ids_)

    print("Building shared feature arrays (genre profile, popularity, activity)...")
    user_genre_profile = build_user_genre_profile(train, user_to_idx, catalog_movie_to_idx, genre_matrix)
    item_popularity = build_item_popularity(train, features["catalog_movie_ids"])
    user_activity = build_user_activity(train, model.user_ids_)

    item_counts = train.groupby("movieId").size()
    decile_items = top_decile_items(item_counts)

    def candidate_features_for(user_ids: np.ndarray) -> pd.DataFrame:
        candidates = model.recommend_batch_with_scores(list(user_ids), CANDIDATE_N)
        return build_candidate_features(
            candidates, user_genre_profile, user_activity, item_popularity,
            genre_matrix, has_genome, user_to_idx, catalog_movie_to_idx,
        )

    def before_rerank_recs(feature_df: pd.DataFrame) -> dict:
        # raw two-tower ordering, truncated from the same top-CANDIDATE_N pool
        # the ranker sees -- isolates the ranker's marginal effect from the
        # larger candidate pool itself.
        ranked = feature_df.sort_values(["user_id", "candidate_rank"])
        return {
            user_id: group["movie_id"].head(TOP_K).tolist()
            for user_id, group in ranked.groupby("user_id")
        }

    # --- split val users: ranker trains on 80%, is scored on the other 20% ---
    val_users = val["userId"].unique()
    warm_val_users = np.array(sorted(set(val_users) & set(user_to_idx)))
    rng = np.random.default_rng(RANDOM_SEED)
    shuffled = rng.permutation(warm_val_users)
    split_point = int(len(shuffled) * (1 - HOLDOUT_FRACTION))
    ranker_train_users, holdout_users = shuffled[:split_point], shuffled[split_point:]
    relevant_val = build_relevant(val, warm_val_users, item_universe)

    print(f"Generating top-{CANDIDATE_N} candidates for {len(ranker_train_users):,} ranker-train users...")
    train_feature_df = candidate_features_for(ranker_train_users)
    train_labels = label_candidates(train_feature_df, relevant_val)

    print(f"Fitting ranker on {len(train_feature_df):,} candidate rows "
          f"(positive rate {train_labels.mean():.4f})...")
    ranker = HistGradientBoostingClassifier(random_state=RANDOM_SEED)
    ranker.fit(train_feature_df[FEATURE_COLUMNS], train_labels)

    print(f"Generating top-{CANDIDATE_N} candidates for {len(holdout_users):,} holdout users...")
    holdout_feature_df = candidate_features_for(holdout_users)
    holdout_relevant = {u: relevant_val[u] for u in holdout_users if u in relevant_val}
    holdout_result = {
        "n_users": len(holdout_users),
        "before_rerank": eval_slice(
            before_rerank_recs(holdout_feature_df), holdout_relevant, item_universe, decile_items
        ),
        "after_rerank": eval_slice(
            rerank(ranker, holdout_feature_df, TOP_K), holdout_relevant, item_universe, decile_items
        ),
    }
    print("\nHoldout (val users the ranker never trained on):")
    print(json.dumps(holdout_result, indent=2))

    # --- final, single-use check against test.parquet ---
    warm_test_users = np.array(sorted(set(test["userId"].unique()) & set(user_to_idx)))
    relevant_test = build_relevant(test, warm_test_users, item_universe)
    print(f"\nGenerating top-{CANDIDATE_N} candidates for {len(warm_test_users):,} test users...")
    test_feature_df = candidate_features_for(warm_test_users)
    test_result = {
        "n_users": len(warm_test_users),
        "before_rerank": eval_slice(
            before_rerank_recs(test_feature_df), relevant_test, item_universe, decile_items
        ),
        "after_rerank": eval_slice(
            rerank(ranker, test_feature_df, TOP_K), relevant_test, item_universe, decile_items
        ),
    }
    print("\nTest set:")
    print(json.dumps(test_result, indent=2))

    result = {
        "candidate_n": CANDIDATE_N,
        "ranker_train_rows": len(train_feature_df),
        "positive_rate": round(float(train_labels.mean()), 4),
        "holdout": holdout_result,
        "test_set": test_result,
    }
    with open(ROOT_DIR / "docs" / "ranker_eval.json", "w") as f:
        json.dump(result, f, indent=2)

    joblib.dump({"model": ranker, "feature_columns": FEATURE_COLUMNS}, RANKER_CHECKPOINT_PATH)
    print(f"\nSaved ranker to {RANKER_CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
