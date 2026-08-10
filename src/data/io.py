import pandas as pd

from src.data.config import LINKS_PATH, MOVIES_PATH, RATINGS_PATH, TAGS_PATH


def load_ratings() -> pd.DataFrame:
    return pd.read_csv(
        RATINGS_PATH,
        dtype={"userId": "int32", "movieId": "int32", "rating": "float32"},
    )


def load_movies() -> pd.DataFrame:
    return pd.read_csv(
        MOVIES_PATH,
        dtype={"movieId": "int32", "title": "string", "genres": "string"},
    )


def load_links() -> pd.DataFrame:
    return pd.read_csv(
        LINKS_PATH,
        dtype={"movieId": "int32", "imdbId": "string", "tmdbId": "string"},
    )


def load_tags() -> pd.DataFrame:
    return pd.read_csv(
        TAGS_PATH,
        dtype={"userId": "int32", "movieId": "int32", "tag": "string"},
    )
