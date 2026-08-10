import pandas as pd


def time_based_split(
    ratings: pd.DataFrame, val_fraction: float, test_fraction: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the interaction stream on a single global timeline.

    This is deliberately NOT a per-user or random split. A global cutoff means
    train only ever contains interactions that happened before val/test in
    wall-clock time, matching how the system will be evaluated in production
    (predict the future from the past). It also means some val/test users and
    movies won't appear in train at all -- that's real cold-start, not a bug,
    and gets reported separately rather than filtered away.
    """
    df = ratings.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    test_cut = int(n * (1 - test_fraction))
    val_cut = int(n * (1 - test_fraction - val_fraction))

    train = df.iloc[:val_cut].reset_index(drop=True)
    val = df.iloc[val_cut:test_cut].reset_index(drop=True)
    test = df.iloc[test_cut:].reset_index(drop=True)
    return train, val, test


def split_report(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> dict:
    train_users, train_movies = set(train["userId"]), set(train["movieId"])

    def cold_start_stats(split: pd.DataFrame) -> dict:
        users = set(split["userId"])
        movies = set(split["movieId"])
        cold_users = users - train_users
        cold_movies = movies - train_movies
        cold_interaction_mask = (~split["userId"].isin(train_users)) | (
            ~split["movieId"].isin(train_movies)
        )
        return {
            "rows": len(split),
            "start_date": pd.to_datetime(split["timestamp"].min(), unit="s").date().isoformat(),
            "end_date": pd.to_datetime(split["timestamp"].max(), unit="s").date().isoformat(),
            "unique_users": len(users),
            "unique_movies": len(movies),
            "cold_start_users": len(cold_users),
            "cold_start_movies": len(cold_movies),
            "rows_touching_cold_start": int(cold_interaction_mask.sum()),
        }

    return {
        "train": {
            "rows": len(train),
            "start_date": pd.to_datetime(train["timestamp"].min(), unit="s").date().isoformat(),
            "end_date": pd.to_datetime(train["timestamp"].max(), unit="s").date().isoformat(),
            "unique_users": len(train_users),
            "unique_movies": len(train_movies),
        },
        "val": cold_start_stats(val),
        "test": cold_start_stats(test),
    }
