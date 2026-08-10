from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT_DIR / "Data" / "ml-25m"
PROCESSED_DIR = ROOT_DIR / "Data" / "processed"

RATINGS_PATH = RAW_DIR / "ratings.csv"
MOVIES_PATH = RAW_DIR / "movies.csv"
LINKS_PATH = RAW_DIR / "links.csv"
TAGS_PATH = RAW_DIR / "tags.csv"
GENOME_SCORES_PATH = RAW_DIR / "genome-scores.csv"
GENOME_TAGS_PATH = RAW_DIR / "genome-tags.csv"

# Fraction of the (time-ordered) interaction stream held out for validation and test.
# Both cuts are made on the global timeline, not per-user, so the split mirrors how the
# system will actually be evaluated in production: train on the past, predict the future.
VAL_FRACTION = 0.10
TEST_FRACTION = 0.10

RANDOM_SEED = 42
