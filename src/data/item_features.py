"""Per-movie content features for the two-tower item tower: a genre
multi-hot vector (covers all movies) and a tag-genome relevance vector
(covers only ~22% of movies -- the rest get a zero vector plus an explicit
has_genome flag, so the model can learn to discount a missing vector rather
than mistake it for "no relevant tags").

These features are what let the item tower score a movie it has never seen
a rating for, which plain ID-embedding models structurally cannot do.

Usage:
    python -m src.data.item_features
"""

import numpy as np
import pandas as pd

from src.data.config import GENOME_SCORES_PATH, PROCESSED_DIR


def build_genre_features(movies: pd.DataFrame) -> pd.DataFrame:
    genre_dummies = movies["genres"].str.get_dummies(sep="|").astype(np.float32)
    return pd.concat([movies[["movieId"]].reset_index(drop=True), genre_dummies], axis=1)


def build_genome_features(genome_scores_path=GENOME_SCORES_PATH) -> tuple[np.ndarray, np.ndarray]:
    scores = pd.read_csv(genome_scores_path)
    pivot = scores.pivot(index="movieId", columns="tagId", values="relevance")
    movie_ids = pivot.index.to_numpy(dtype=np.int32)
    matrix = pivot.to_numpy(dtype=np.float32)
    return movie_ids, matrix


def main() -> None:
    movies = pd.read_parquet(PROCESSED_DIR / "movies.parquet")

    genre_df = build_genre_features(movies)
    genre_df.to_parquet(PROCESSED_DIR / "item_genre_features.parquet", index=False)
    genre_cols = [c for c in genre_df.columns if c != "movieId"]
    print(f"genre features: {len(genre_df):,} movies x {len(genre_cols)} genres -> {genre_cols}")

    genome_ids, genome_matrix = build_genome_features()
    np.save(PROCESSED_DIR / "item_genome_movie_ids.npy", genome_ids)
    np.save(PROCESSED_DIR / "item_genome_features.npy", genome_matrix)
    print(
        f"genome features: {genome_matrix.shape[0]:,} movies x {genome_matrix.shape[1]} tags "
        f"({100 * genome_matrix.shape[0] / len(movies):.1f}% of catalog)"
    )


if __name__ == "__main__":
    main()
