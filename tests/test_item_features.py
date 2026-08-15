import pandas as pd

from src.data.item_features import build_genre_features


def test_genre_multihot_encoding():
    movies = pd.DataFrame(
        {
            "movieId": [1, 2, 3],
            "title": ["A", "B", "C"],
            "genres": ["Comedy|Drama", "Comedy", "(no genres listed)"],
        }
    )
    result = build_genre_features(movies)
    assert set(result.columns) == {"movieId", "Comedy", "Drama", "(no genres listed)"}
    row1 = result.loc[result["movieId"] == 1].iloc[0]
    assert row1["Comedy"] == 1 and row1["Drama"] == 1 and row1["(no genres listed)"] == 0
    row3 = result.loc[result["movieId"] == 3].iloc[0]
    assert row3["(no genres listed)"] == 1 and row3["Comedy"] == 0
