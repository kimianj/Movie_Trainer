"""Build the processed, time-split MovieLens dataset from raw CSVs.

"""

import json

from src.data.clean import clean_ratings
from src.data.config import (
    PROCESSED_DIR,
    RANDOM_SEED,
    TEST_FRACTION,
    VAL_FRACTION,
)
from src.data.io import load_movies, load_ratings
from src.data.split import split_report, time_based_split


def main() -> None:
    print("Loading raw ratings and movies...")
    ratings = load_ratings()
    movies = load_movies()
    print(f"  raw ratings: {len(ratings):,} rows, {len(movies):,} movies")

    print("Cleaning ratings...")
    clean, clean_report = clean_ratings(ratings, movies)
    print(json.dumps(clean_report, indent=2))

    print(f"Splitting on global timeline (val={VAL_FRACTION}, test={TEST_FRACTION})...")
    train, val, test = time_based_split(clean, VAL_FRACTION, TEST_FRACTION)
    report = split_report(train, val, test)
    print(json.dumps(report, indent=2))

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train.to_parquet(PROCESSED_DIR / "train.parquet", index=False)
    val.to_parquet(PROCESSED_DIR / "val.parquet", index=False)
    test.to_parquet(PROCESSED_DIR / "test.parquet", index=False)
    movies.to_parquet(PROCESSED_DIR / "movies.parquet", index=False)

    with open(PROCESSED_DIR / "split_report.json", "w") as f:
        json.dump({"clean": clean_report, "split": report, "seed": RANDOM_SEED}, f, indent=2)

    print(f"Wrote processed dataset to {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
