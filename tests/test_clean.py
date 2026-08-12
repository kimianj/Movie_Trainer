import pandas as pd

from src.data.clean import clean_ratings

MOVIES = pd.DataFrame({"movieId": [1, 2, 3], "title": ["A", "B", "C"], "genres": ["x", "y", "z"]})


def _ratings(rows):
    return pd.DataFrame(rows, columns=["userId", "movieId", "rating", "timestamp"])


def test_drops_exact_duplicate_rows():
    df = _ratings([[1, 1, 4.0, 100], [1, 1, 4.0, 100]])
    clean, report = clean_ratings(df, MOVIES)
    assert len(clean) == 1
    assert report["exact_duplicate_rows_dropped"] == 1


def test_rerating_keeps_most_recent():
    df = _ratings([[1, 1, 2.0, 100], [1, 1, 5.0, 200]])
    clean, report = clean_ratings(df, MOVIES)
    assert len(clean) == 1
    assert clean.iloc[0]["rating"] == 5.0
    assert report["stale_rerating_rows_dropped"] == 1


def test_drops_invalid_rating_values():
    df = _ratings([[1, 1, 4.0, 100], [1, 2, 4.3, 101]])
    clean, report = clean_ratings(df, MOVIES)
    assert len(clean) == 1
    assert report["invalid_rating_value_rows_dropped"] == 1


def test_drops_nonpositive_timestamps():
    df = _ratings([[1, 1, 4.0, 100], [1, 2, 4.0, 0], [1, 3, 4.0, -5]])
    clean, report = clean_ratings(df, MOVIES)
    assert len(clean) == 1
    assert report["invalid_timestamp_rows_dropped"] == 2


def test_drops_orphan_movie_ids():
    df = _ratings([[1, 1, 4.0, 100], [1, 999, 4.0, 101]])
    clean, report = clean_ratings(df, MOVIES)
    assert len(clean) == 1
    assert report["orphan_movie_id_rows_dropped"] == 1


def test_output_sorted_by_timestamp():
    df = _ratings([[1, 1, 4.0, 300], [1, 2, 4.0, 100], [1, 3, 4.0, 200]])
    clean, _ = clean_ratings(df, MOVIES)
    assert list(clean["timestamp"]) == [100, 200, 300]
