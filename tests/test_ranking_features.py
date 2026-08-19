import math

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier

from src.ranking.features import (
    FEATURE_COLUMNS,
    build_candidate_features,
    build_item_popularity,
    build_user_activity,
    build_user_genre_profile,
    label_candidates,
    rerank,
)


def test_build_user_genre_profile_averages_correctly():
    train = pd.DataFrame({"userId": [1, 1, 2], "movieId": [10, 20, 10]})
    user_to_idx = {1: 1, 2: 2}
    catalog_movie_to_idx = {10: 0, 20: 1, 30: 2}
    genre_matrix = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32)

    profile = build_user_genre_profile(train, user_to_idx, catalog_movie_to_idx, genre_matrix)

    assert profile.shape == (2, 2)
    assert profile[0] == pytest.approx([0.5, 0.5])  # user 1 rated movies 10 and 20
    assert profile[1] == pytest.approx([1.0, 0.0])  # user 2 rated only movie 10


def test_build_item_popularity_log1p_and_zero_for_unseen_items():
    train = pd.DataFrame({"movieId": [10, 10, 20]})
    catalog_movie_ids = np.array([10, 20, 30])

    popularity = build_item_popularity(train, catalog_movie_ids)

    assert popularity == pytest.approx([math.log1p(2), math.log1p(1), 0.0])


def test_build_user_activity_log1p_and_zero_for_unseen_users():
    train = pd.DataFrame({"userId": [1, 1, 2]})
    user_ids = np.array([1, 2, 3])

    activity = build_user_activity(train, user_ids)

    assert activity == pytest.approx([math.log1p(2), math.log1p(1), 0.0])


def test_build_candidate_features_produces_expected_columns_and_row_count():
    candidates = {1: [(10, 0.9), (20, 0.5)]}
    user_genre_profile = np.array([[0.5, 0.5]], dtype=np.float32)
    user_activity = np.array([1.0], dtype=np.float32)
    item_popularity = np.array([2.0, 1.0, 0.0], dtype=np.float32)
    genre_matrix = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32)
    has_genome = np.array([1.0, 0.0, 1.0], dtype=np.float32)
    user_to_idx = {1: 1}
    catalog_movie_to_idx = {10: 0, 20: 1, 30: 2}

    df = build_candidate_features(
        candidates, user_genre_profile, user_activity, item_popularity,
        genre_matrix, has_genome, user_to_idx, catalog_movie_to_idx,
    )

    assert len(df) == 2
    assert set(FEATURE_COLUMNS) <= set(df.columns)
    first = df.iloc[0]
    assert first["movie_id"] == 10
    assert first["candidate_rank"] == 1
    assert first["two_tower_score"] == pytest.approx(0.9)
    assert first["log_item_popularity"] == pytest.approx(2.0)
    assert first["genre_overlap"] == pytest.approx(0.5)  # dot([0.5,0.5],[1,0])
    assert first["has_genome"] == pytest.approx(1.0)
    assert first["log_user_activity"] == pytest.approx(1.0)


def test_label_candidates_marks_relevant_items_positive_others_negative():
    feature_df = pd.DataFrame({"user_id": [1, 1, 2], "movie_id": [10, 20, 10]})
    relevant = {1: {20}, 2: set()}

    labels = label_candidates(feature_df, relevant)

    assert labels.tolist() == [0, 1, 0]


class _StubModel:
    def __init__(self, positive_scores):
        self.positive_scores = positive_scores

    def predict_proba(self, X):
        scores = np.array(self.positive_scores)
        return np.column_stack([1 - scores, scores])


def test_rerank_sorts_by_predicted_score_and_truncates_to_k():
    feature_df = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2],
            "movie_id": [10, 20, 30, 40],
            **{col: [0.0] * 4 for col in FEATURE_COLUMNS},
        }
    )
    model = _StubModel(positive_scores=[0.2, 0.9, 0.5, 0.1])

    result = rerank(model, feature_df, k=2)

    assert result[1] == [20, 30]  # scores 0.9, 0.5 beat 0.2
    assert result[2] == [40]


def test_ranker_end_to_end_on_tiny_synthetic_data():
    rng = np.random.default_rng(0)
    n = 20
    feature_df = pd.DataFrame(
        {
            "user_id": [1] * 10 + [2] * 10,
            "movie_id": list(range(100, 110)) + list(range(200, 210)),
            "two_tower_score": rng.normal(size=n),
            "candidate_rank": list(range(1, 11)) * 2,
            "log_item_popularity": rng.normal(size=n),
            "genre_overlap": rng.normal(size=n),
            "has_genome": rng.integers(0, 2, size=n).astype(float),
            "log_user_activity": rng.normal(size=n),
        }
    )
    labels = pd.Series([i % 2 for i in range(n)])  # guarantees both classes present

    model = HistGradientBoostingClassifier(random_state=0).fit(feature_df[FEATURE_COLUMNS], labels)
    result = rerank(model, feature_df, k=3)

    assert set(result.keys()) == {1, 2}
    assert all(len(v) <= 3 for v in result.values())
