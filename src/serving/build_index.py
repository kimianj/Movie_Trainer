"""Builds the artifacts the FastAPI serving layer needs to answer
/recommend requests without loading any of the large, training-only raw
data at request time: a FAISS index over the full-catalog item embeddings,
the ranking-stage feature arrays (computed once here and reused verbatim by
both src/ranking/train.py and api/recommender.py -- one implementation, not
two, so there's no train/serve skew), and a compact seen-items mask so
already-rated items get excluded from recommendations.

Exact search (IndexFlatIP), not approximate (IndexIVFFlat): the latency
benchmark in src/models/compare.py already found brute-force/exact search
is sub-millisecond at this catalog's real scale (tens of thousands of
items), and IndexIVFFlat's approximate results only overlap with exact
17-72% of the time depending on scale -- not worth trading accuracy for a
latency win nobody needs yet.

The 59MB tag-genome feature matrix is loaded here (to compute the trained
item embeddings) and nowhere else downstream -- only the small has_genome
flag it produces gets written for serving, keeping the servable artifact
set small.

Usage:
    python -m src.serving.build_index
"""

import faiss
import numpy as np
import pandas as pd
from scipy import sparse

from src.data.config import PROCESSED_DIR, SERVING_DIR
from src.models.two_tower import TwoTowerRecommender, load_catalog_features, load_checkpoint
from src.ranking.features import build_item_popularity, build_user_activity, build_user_genre_profile


def main() -> None:
    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")

    print("Loading checkpoint + catalog features...")
    features = load_catalog_features()
    user_tower, item_tower, user_to_idx, item_to_idx, features = load_checkpoint(features=features)
    model = TwoTowerRecommender(user_tower, item_tower, user_to_idx, item_to_idx, features, train)

    SERVING_DIR.mkdir(parents=True, exist_ok=True)

    print("Building FAISS exact-search index over full-catalog item embeddings...")
    item_vectors = np.ascontiguousarray(model.item_embeddings_, dtype=np.float32)
    index = faiss.IndexFlatIP(item_vectors.shape[1])
    index.add(item_vectors)
    faiss.write_index(index, str(SERVING_DIR / "item_index.faiss"))

    np.save(SERVING_DIR / "catalog_movie_ids.npy", model.catalog_movie_ids)
    genre_matrix = features["genre_matrix"].numpy()
    np.save(SERVING_DIR / "genre_matrix.npy", genre_matrix)
    np.save(SERVING_DIR / "has_genome.npy", features["has_genome"].numpy())

    print("Building seen-items mask (bool dtype, compressed)...")
    sparse.save_npz(SERVING_DIR / "seen_items.npz", model.X_.astype(bool), compressed=True)

    print("Building ranking feature arrays (reusing src.ranking.features)...")
    np.save(
        SERVING_DIR / "item_popularity.npy",
        build_item_popularity(train, features["catalog_movie_ids"]),
    )
    np.save(
        SERVING_DIR / "user_genre_profile.npy",
        build_user_genre_profile(train, user_to_idx, features["catalog_movie_to_idx"], genre_matrix),
    )
    np.save(SERVING_DIR / "user_activity.npy", build_user_activity(train, model.user_ids_))

    print("\nServing artifacts:")
    total_mb = 0.0
    for path in sorted(SERVING_DIR.iterdir()):
        size_mb = path.stat().st_size / 1e6
        total_mb += size_mb
        print(f"  {path.name}: {size_mb:.2f} MB")
    checkpoint_mb = (PROCESSED_DIR / "two_tower_checkpoint.pt").stat().st_size / 1e6
    ranker_path = PROCESSED_DIR / "ranker_model.joblib"
    ranker_mb = ranker_path.stat().st_size / 1e6 if ranker_path.exists() else 0.0
    print(f"  ../two_tower_checkpoint.pt: {checkpoint_mb:.2f} MB")
    print(f"  ../ranker_model.joblib: {ranker_mb:.2f} MB")
    print(f"Total servable footprint: {total_mb + checkpoint_mb + ranker_mb:.2f} MB")


if __name__ == "__main__":
    main()
