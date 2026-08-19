"""FastAPI serving layer: loads the trained two-tower user tower, FAISS
item index, and ranking model once at startup, then answers /recommend
requests with FAISS retrieval -> seen-item exclusion -> ranking rerank.

If the serving artifacts haven't been built yet (python -m src.serving.build_index
and python -m src.ranking.train, which write to Data/processed/, gitignored
and not present in a fresh checkout), startup degrades gracefully instead of
crashing: /health reports "degraded" and /recommend returns 503. This is
what keeps this app importable and testable in CI, which has no Data/ at
all -- see tests/test_api.py.

Usage:
    uvicorn api.main:app --reload
"""

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException

from src.data.config import PROCESSED_DIR

from api.recommender import RecommenderService
from api.schemas import HealthResponse, RecommendResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.recommender = RecommenderService.load(PROCESSED_DIR)
    except FileNotFoundError:
        app.state.recommender = None
    yield


app = FastAPI(title="Movie Recommender API", lifespan=lifespan)


def get_recommender() -> RecommenderService:
    recommender = app.state.recommender
    if recommender is None:
        raise HTTPException(503, "Recommender artifacts not loaded")
    return recommender


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    recommender = app.state.recommender
    return HealthResponse(
        status="ok" if recommender else "degraded",
        artifacts_loaded=recommender is not None,
        catalog_size=len(recommender.artifacts.catalog_movie_ids) if recommender else None,
    )


@app.get("/recommend/{user_id}", response_model=RecommendResponse)
def recommend(
    user_id: int, k: int = 10, recommender: RecommenderService = Depends(get_recommender)
) -> RecommendResponse:
    if not 1 <= k <= recommender.candidate_n:
        raise HTTPException(400, f"k must be between 1 and {recommender.candidate_n}")
    return RecommendResponse(
        user_id=user_id,
        personalized=recommender.is_known_user(user_id),
        recommendations=recommender.recommend(user_id, k),
    )
