"""Exploratory analysis on the cleaned dataset: sparsity, rating distribution,
popularity long-tail, and the timeline density behind the cold-start numbers
from the split report.
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.data.config import PROCESSED_DIR, ROOT_DIR

FIGURES_DIR = ROOT_DIR / "docs" / "figures"


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    val = pd.read_parquet(PROCESSED_DIR / "val.parquet")
    test = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    all_ratings = pd.concat([train, val, test], ignore_index=True)

    n_users = all_ratings["userId"].nunique()
    n_movies = all_ratings["movieId"].nunique()
    n_ratings = len(all_ratings)
    density = n_ratings / (n_users * n_movies)

    summary = {
        "n_users": int(n_users),
        "n_movies": int(n_movies),
        "n_ratings": int(n_ratings),
        "matrix_density_pct": round(density * 100, 4),
        "rating_mean": round(float(all_ratings["rating"].mean()), 3),
        "rating_std": round(float(all_ratings["rating"].std()), 3),
    }

    # Ratings per user / per movie -- long-tail check.
    ratings_per_user = all_ratings.groupby("userId").size()
    ratings_per_movie = all_ratings.groupby("movieId").size()
    summary["ratings_per_user"] = {
        "min": int(ratings_per_user.min()),
        "median": int(ratings_per_user.median()),
        "mean": round(float(ratings_per_user.mean()), 1),
        "p95": int(ratings_per_user.quantile(0.95)),
        "max": int(ratings_per_user.max()),
    }
    summary["ratings_per_movie"] = {
        "min": int(ratings_per_movie.min()),
        "median": int(ratings_per_movie.median()),
        "mean": round(float(ratings_per_movie.mean()), 1),
        "p95": int(ratings_per_movie.quantile(0.95)),
        "max": int(ratings_per_movie.max()),
    }
    top_decile_movie_share = (
        ratings_per_movie.sort_values(ascending=False)
        .head(max(1, len(ratings_per_movie) // 10))
        .sum()
        / n_ratings
    )
    summary["top_10pct_movies_share_of_ratings_pct"] = round(top_decile_movie_share * 100, 1)

    # Timeline density: ratings per day, to explain the split report's cold-start spike.
    dates = pd.to_datetime(all_ratings["timestamp"], unit="s").dt.date
    ratings_per_day = dates.value_counts().sort_index()
    summary["busiest_days"] = {
        str(d): int(c) for d, c in ratings_per_day.sort_values(ascending=False).head(5).items()
    }
    summary["median_ratings_per_day"] = int(ratings_per_day.median())

    # First-appearance date per movie -- are val/test cold-start movies new titles
    # (real catalog growth) or old titles that just weren't rated yet in train?
    first_rating = all_ratings.groupby("movieId")["timestamp"].min()
    first_rating_date = pd.to_datetime(first_rating, unit="s")
    val_test_movies = set(val["movieId"]) | set(test["movieId"])
    train_movies = set(train["movieId"])
    cold_movies = val_test_movies - train_movies
    cold_first_seen = first_rating_date.loc[list(cold_movies)]
    summary["cold_start_movies_first_rated_after_2016-06-25_pct"] = round(
        100 * (cold_first_seen > "2016-06-25").mean(), 1
    )

    with open(ROOT_DIR / "docs" / "eda_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    # --- Figures ---
    fig, ax = plt.subplots(figsize=(6, 4))
    all_ratings["rating"].value_counts().sort_index().plot(kind="bar", ax=ax)
    ax.set_title("Rating value distribution")
    ax.set_xlabel("rating")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "rating_distribution.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ratings_per_day.plot(ax=ax)
    ax.set_title("Ratings per day over time (full 1995-2019 span)")
    ax.set_ylabel("ratings/day")
    ax.axvline(pd.Timestamp("2016-06-25"), color="red", linestyle="--", label="train/val cut")
    ax.axvline(pd.Timestamp("2018-01-04"), color="orange", linestyle="--", label="val/test cut")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "ratings_per_day.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ratings_per_movie.sort_values(ascending=False).reset_index(drop=True).plot(ax=ax)
    ax.set_yscale("log")
    ax.set_title("Ratings per movie, sorted descending (log scale)")
    ax.set_xlabel("movie rank")
    ax.set_ylabel("num ratings (log)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "movie_popularity_long_tail.png", dpi=120)
    plt.close(fig)

    print(f"\nFigures written to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
