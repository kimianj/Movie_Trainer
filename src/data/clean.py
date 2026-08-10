import pandas as pd


def clean_ratings(ratings: pd.DataFrame, movies: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Deduplicate and validate the raw ratings stream.

    Returns the cleaned dataframe plus a report of what was removed, so the
    numbers can be cited in the EDA writeup instead of silently discarded.
    """
    report = {"raw_rows": len(ratings)}

    df = ratings.drop_duplicates()
    report["exact_duplicate_rows_dropped"] = report["raw_rows"] - len(df)

    # A user can appear twice for the same movie if they re-rated it; keep the
    # most recent rating since that's the signal that would actually be observed
    # by the system at serving time.
    before = len(df)
    df = df.sort_values("timestamp").drop_duplicates(
        subset=["userId", "movieId"], keep="last"
    )
    report["stale_rerating_rows_dropped"] = before - len(df)

    valid_ratings = {i * 0.5 for i in range(1, 11)}
    bad_rating_mask = ~df["rating"].isin(valid_ratings)
    report["invalid_rating_value_rows_dropped"] = int(bad_rating_mask.sum())
    df = df[~bad_rating_mask]

    bad_timestamp_mask = df["timestamp"] <= 0
    report["invalid_timestamp_rows_dropped"] = int(bad_timestamp_mask.sum())
    df = df[~bad_timestamp_mask]

    known_movie_ids = set(movies["movieId"])
    orphan_mask = ~df["movieId"].isin(known_movie_ids)
    report["orphan_movie_id_rows_dropped"] = int(orphan_mask.sum())
    df = df[~orphan_mask]

    report["clean_rows"] = len(df)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df, report
