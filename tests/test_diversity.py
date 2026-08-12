import pandas as pd
import pytest

from src.eval.diversity import catalog_coverage, popularity_bias, top_decile_items


def test_catalog_coverage_fraction_of_universe_recommended():
    recs = {1: [1, 2], 2: [2, 3]}
    assert catalog_coverage(recs, {1, 2, 3, 4}) == pytest.approx(0.75)


def test_catalog_coverage_empty_universe_is_nan():
    import math

    assert math.isnan(catalog_coverage({}, set()))


def test_popularity_bias_averages_per_user_share():
    recs = {1: [1, 2], 2: [3, 4]}
    assert popularity_bias(recs, top_decile_items={1, 3}) == pytest.approx(0.5)


def test_popularity_bias_all_popular_is_one():
    recs = {1: [1, 2]}
    assert popularity_bias(recs, top_decile_items={1, 2}) == pytest.approx(1.0)


def test_top_decile_items_picks_highest_counts():
    counts = pd.Series({i: 100 - i for i in range(1, 11)})  # movieId 1 has the highest count
    result = top_decile_items(counts)
    assert result == {1}
