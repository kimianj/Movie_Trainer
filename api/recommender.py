"""Wraps loaded serving artifacts into the recommend() call the FastAPI
routes use. RecommenderService.load() is the only disk-touching path --
the dataclass itself takes an already-loaded ServingArtifacts and does no
I/O, so tests build one directly from small in-memory fakes instead of
requiring the real trained model on disk (see tests/test_api.py).
"""

from dataclasses import dataclass, field

import numpy as np
import torch

from src.ranking.features import CANDIDATE_N, FEATURE_COLUMNS, build_candidate_features
from src.serving.loader import ServingArtifacts, load_serving_artifacts

from api.schemas import RecommendationItem


@dataclass
class RecommenderService:
    artifacts: ServingArtifacts
    candidate_n: int = CANDIDATE_N
    _title_by_movie_id: dict = field(init=False, repr=False)
    _catalog_movie_to_idx: dict = field(init=False, repr=False)

    def __post_init__(self):
        self._title_by_movie_id = dict(
            zip(self.artifacts.movies["movieId"], self.artifacts.movies["title"])
        )
        self._catalog_movie_to_idx = {m: i for i, m in enumerate(self.artifacts.catalog_movie_ids)}

    @classmethod
    def load(cls, processed_dir) -> "RecommenderService":
        return cls(artifacts=load_serving_artifacts(processed_dir))

    def is_known_user(self, user_id: int) -> bool:
        return user_id in self.artifacts.user_to_idx

    def recommend(self, user_id: int, k: int) -> list[RecommendationItem]:
        idx = self.artifacts.user_to_idx.get(user_id, 0)  # 0 = UNK
        with torch.no_grad():
            user_vec = self.artifacts.user_tower(torch.tensor([idx])).numpy()
        user_vec = np.ascontiguousarray(user_vec, dtype=np.float32)

        scores, catalog_idx = self.artifacts.item_index.search(user_vec, self.candidate_n)
        scores, catalog_idx = scores[0], catalog_idx[0]

        if idx != 0:
            seen_row = self.artifacts.seen_items[idx - 1].toarray()[0]
            keep = ~seen_row[catalog_idx]
            scores, catalog_idx = scores[keep], catalog_idx[keep]

        movie_ids = self.artifacts.catalog_movie_ids[catalog_idx]

        if idx == 0:
            # Unknown user: no train history to build ranking features from,
            # so serve the two-tower's own top-k content-similarity ranking
            # directly instead of reranking with fabricated zero-signal
            # features -- the same UNK-embedding cold-start fallback the
            # model was explicitly trained to support.
            top_ids = movie_ids[:k].tolist()
            top_scores = scores[:k].tolist()
        else:
            candidates = {user_id: list(zip(movie_ids.tolist(), scores.tolist()))}
            feature_df = build_candidate_features(
                candidates,
                self.artifacts.user_genre_profile,
                self.artifacts.user_activity,
                self.artifacts.item_popularity,
                self.artifacts.genre_matrix,
                self.artifacts.has_genome,
                self.artifacts.user_to_idx,
                self._catalog_movie_to_idx,
            )
            predicted = self.artifacts.ranker_model.predict_proba(feature_df[FEATURE_COLUMNS])[:, 1]
            top = feature_df.assign(predicted_score=predicted).sort_values(
                "predicted_score", ascending=False
            ).head(k)
            top_ids = top["movie_id"].tolist()
            top_scores = top["predicted_score"].tolist()

        return [
            RecommendationItem(
                movie_id=int(movie_id),
                title=self._title_by_movie_id.get(int(movie_id), "Unknown title"),
                score=float(score),
            )
            for movie_id, score in zip(top_ids, top_scores)
        ]
