from pydantic import BaseModel


class RecommendationItem(BaseModel):
    movie_id: int
    title: str
    score: float


class RecommendResponse(BaseModel):
    user_id: int
    # False when user_id is unknown to the trained model and was served via
    # the UNK embedding row -- the same user-cold-start limit the README
    # documents (MovieLens has no user side-info to fall back on).
    personalized: bool
    recommendations: list[RecommendationItem]


class HealthResponse(BaseModel):
    status: str  # "ok" | "degraded"
    artifacts_loaded: bool
    catalog_size: int | None = None
