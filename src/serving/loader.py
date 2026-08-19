"""Thin I/O wrapper that loads the serving artifacts src/serving/build_index.py
and src/ranking/train.py write -- everything api/recommender.py needs to
answer a /recommend request without touching raw training data or the
(training-only) tag-genome features at request time. Not directly unit
tested, same convention as load_catalog_features()/TwoTowerRecommender in
src/models/two_tower.py -- exercised via the real local run, and bypassed
entirely in tests via dependency injection (api/recommender.py).
"""

from dataclasses import dataclass

import faiss
import joblib
import numpy as np
import pandas as pd
from scipy import sparse

from src.data.config import PROCESSED_DIR
from src.models.two_tower import UserTower, load_user_tower


@dataclass
class ServingArtifacts:
    user_tower: UserTower
    user_to_idx: dict
    item_index: faiss.Index
    catalog_movie_ids: np.ndarray
    genre_matrix: np.ndarray
    has_genome: np.ndarray
    item_popularity: np.ndarray
    user_genre_profile: np.ndarray
    user_activity: np.ndarray
    seen_items: sparse.csr_matrix
    ranker_model: object
    ranker_feature_columns: list
    movies: pd.DataFrame


def load_serving_artifacts(processed_dir=PROCESSED_DIR) -> ServingArtifacts:
    serving_dir = processed_dir / "serving"
    index_path = serving_dir / "item_index.faiss"
    if not index_path.exists():
        raise FileNotFoundError(index_path)

    user_tower, user_to_idx = load_user_tower(processed_dir / "two_tower_checkpoint.pt")
    ranker_bundle = joblib.load(processed_dir / "ranker_model.joblib")

    return ServingArtifacts(
        user_tower=user_tower,
        user_to_idx=user_to_idx,
        item_index=faiss.read_index(str(index_path)),
        catalog_movie_ids=np.load(serving_dir / "catalog_movie_ids.npy"),
        genre_matrix=np.load(serving_dir / "genre_matrix.npy"),
        has_genome=np.load(serving_dir / "has_genome.npy"),
        item_popularity=np.load(serving_dir / "item_popularity.npy"),
        user_genre_profile=np.load(serving_dir / "user_genre_profile.npy"),
        user_activity=np.load(serving_dir / "user_activity.npy"),
        seen_items=sparse.load_npz(serving_dir / "seen_items.npz"),
        ranker_model=ranker_bundle["model"],
        ranker_feature_columns=ranker_bundle["feature_columns"],
        movies=pd.read_parquet(processed_dir / "movies.parquet"),
    )
