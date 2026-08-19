import faiss
import numpy as np
import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient
from scipy import sparse

import api.main as main_module
from api.main import app, get_recommender
from api.recommender import RecommenderService
from src.data.config import PROCESSED_DIR
from src.models.two_tower import UserTower
from src.ranking.features import FEATURE_COLUMNS
from src.serving.loader import ServingArtifacts


class _FakeRanker:
    """Deterministic stand-in for the real HistGradientBoostingClassifier:
    favors lower candidate_rank, so reranked order is predictable in tests."""

    def predict_proba(self, X):
        prob = 1.0 / X["candidate_rank"].to_numpy()
        return np.column_stack([1 - prob, prob])


def _fake_service(candidate_n: int = 3) -> RecommenderService:
    torch.manual_seed(0)
    catalog_movie_ids = np.array([10, 20, 30, 40, 50])
    genre_matrix = np.array([[1, 0], [0, 1], [1, 1], [0, 0], [1, 0]], dtype=np.float32)
    has_genome = np.array([1, 0, 1, 0, 1], dtype=np.float32)
    item_popularity = np.array([2.0, 1.0, 0.0, 3.0, 1.5], dtype=np.float32)

    user_to_idx = {101: 1, 102: 2}
    user_genre_profile = np.array([[0.5, 0.5], [1.0, 0.0]], dtype=np.float32)
    user_activity = np.array([1.0, 2.0], dtype=np.float32)

    user_tower = UserTower(n_train_users=2, id_dim=4, out_dim=4)
    user_tower.eval()

    item_vectors = np.random.default_rng(0).normal(size=(5, 4)).astype(np.float32)
    index = faiss.IndexFlatIP(4)
    index.add(item_vectors)

    # user 101 (row 0) has already seen movie 10 (catalog position 0).
    seen_items = sparse.csr_matrix(([True], ([0], [0])), shape=(2, 5), dtype=bool)

    movies = pd.DataFrame(
        {"movieId": catalog_movie_ids, "title": [f"Movie {m}" for m in catalog_movie_ids]}
    )

    artifacts = ServingArtifacts(
        user_tower=user_tower,
        user_to_idx=user_to_idx,
        item_index=index,
        catalog_movie_ids=catalog_movie_ids,
        genre_matrix=genre_matrix,
        has_genome=has_genome,
        item_popularity=item_popularity,
        user_genre_profile=user_genre_profile,
        user_activity=user_activity,
        seen_items=seen_items,
        ranker_model=_FakeRanker(),
        ranker_feature_columns=FEATURE_COLUMNS,
        movies=movies,
    )
    return RecommenderService(artifacts=artifacts, candidate_n=candidate_n)


@pytest.fixture
def fake_client():
    app.dependency_overrides[get_recommender] = _fake_service
    app.state.recommender = _fake_service()
    client = TestClient(app)  # no `with`: skips lifespan, never touches disk
    yield client
    app.dependency_overrides.pop(get_recommender, None)


def test_recommend_returns_items_sorted_by_score(fake_client):
    resp = fake_client.get("/recommend/101?k=2")
    assert resp.status_code == 200
    body = resp.json()
    assert body["user_id"] == 101
    assert body["personalized"] is True
    scores = [r["score"] for r in body["recommendations"]]
    assert scores == sorted(scores, reverse=True)
    assert len(body["recommendations"]) <= 2


def test_recommend_unknown_user_falls_back_to_unk_and_flags_personalized_false(fake_client):
    resp = fake_client.get("/recommend/999?k=2")
    body = resp.json()
    assert body["personalized"] is False
    assert len(body["recommendations"]) <= 2


def test_recommend_invalid_k_returns_400(fake_client):
    assert fake_client.get("/recommend/101?k=0").status_code == 400
    assert fake_client.get("/recommend/101?k=100").status_code == 400


def test_health_ok_when_recommender_available(fake_client):
    resp = fake_client.get("/health")
    body = resp.json()
    assert body["status"] == "ok"
    assert body["artifacts_loaded"] is True
    assert body["catalog_size"] == 5


def test_health_degraded_without_artifacts_on_disk(monkeypatch):
    # Point at a directory that doesn't exist, regardless of whether real
    # trained artifacts happen to be present locally -- this is what CI sees
    # (Data/ is gitignored and absent from a fresh checkout).
    monkeypatch.setattr(main_module, "PROCESSED_DIR", PROCESSED_DIR / "__no_such_dir__")
    app.dependency_overrides.pop(get_recommender, None)

    with TestClient(app) as client:  # triggers the real lifespan
        resp = client.get("/health")

    body = resp.json()
    assert body["status"] == "degraded"
    assert body["artifacts_loaded"] is False
