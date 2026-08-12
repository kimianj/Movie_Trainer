import pandas as pd

from src.data.split import split_report, time_based_split


def _ratings(n, users, movies):
    """n rows spread across `users`/`movies`, timestamp == row index (already sorted)."""
    return pd.DataFrame(
        {
            "userId": [users[i % len(users)] for i in range(n)],
            "movieId": [movies[i % len(movies)] for i in range(n)],
            "rating": [4.0] * n,
            "timestamp": list(range(n)),
        }
    )


def test_split_sizes_match_fractions():
    df = _ratings(100, users=list(range(20)), movies=list(range(20)))
    train, val, test = time_based_split(df, val_fraction=0.1, test_fraction=0.1)
    assert len(train) == 80
    assert len(val) == 10
    assert len(test) == 10


def test_no_time_leakage_across_splits():
    df = _ratings(100, users=list(range(20)), movies=list(range(20)))
    train, val, test = time_based_split(df, val_fraction=0.1, test_fraction=0.1)
    assert train["timestamp"].max() <= val["timestamp"].min()
    assert val["timestamp"].max() <= test["timestamp"].min()


def test_split_is_not_shuffled_by_userid_or_movieid():
    # Rows arrive out of timestamp order; the split must still cut on time.
    df = _ratings(100, users=list(range(20)), movies=list(range(20))).sample(
        frac=1.0, random_state=0
    )
    train, val, test = time_based_split(df, val_fraction=0.1, test_fraction=0.1)
    assert len(train) == 80 and len(val) == 10 and len(test) == 10
    assert train["timestamp"].max() <= val["timestamp"].min()


def test_split_report_flags_cold_start():
    # user 99 and movie 99 only ever appear in val, never in train.
    train = pd.DataFrame(
        {"userId": [1, 2, 3], "movieId": [1, 2, 3], "rating": [4.0] * 3, "timestamp": [1, 2, 3]}
    )
    val = pd.DataFrame(
        {"userId": [1, 99], "movieId": [1, 99], "rating": [4.0] * 2, "timestamp": [4, 5]}
    )
    test = pd.DataFrame(
        {"userId": [1], "movieId": [1], "rating": [4.0], "timestamp": [6]}
    )
    report = split_report(train, val, test)
    assert report["val"]["cold_start_users"] == 1
    assert report["val"]["cold_start_movies"] == 1
    assert report["val"]["rows_touching_cold_start"] == 1
