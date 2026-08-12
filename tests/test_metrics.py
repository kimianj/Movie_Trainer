import math

import pytest

from src.eval.metrics import evaluate, ndcg_at_k, recall_at_k


def test_recall_at_k_basic():
    assert recall_at_k([1, 2, 3, 4, 5], {3, 7, 9}, k=5) == pytest.approx(1 / 3)


def test_recall_at_k_only_counts_within_k():
    # relevant item 5 is ranked 6th, outside k=5, so it doesn't count as a hit.
    assert recall_at_k([1, 2, 3, 4, 6, 5], {5}, k=5) == 0.0


def test_recall_at_k_no_relevant_items_is_nan():
    assert math.isnan(recall_at_k([1, 2, 3], set(), k=3))


def test_ndcg_at_k_perfect_rank_is_one():
    assert ndcg_at_k([3, 1, 2], {3}, k=3) == pytest.approx(1.0)


def test_ndcg_at_k_rewards_earlier_hits_more():
    early = ndcg_at_k([3, 1, 2], {3}, k=3)
    late = ndcg_at_k([1, 2, 3], {3}, k=3)
    assert early > late


def test_evaluate_skips_users_with_no_relevant_items():
    recs = {1: [1, 2, 3], 2: [1, 2, 3]}
    relevant = {1: {3}, 2: set()}
    result = evaluate(recs, relevant, k=3)
    assert result["n_users_evaluated"] == 1
    assert result["n_users_skipped_no_relevant_items"] == 1
    assert result["recall_at_k"] == pytest.approx(1.0)


def test_evaluate_missing_user_in_recs_scores_as_zero():
    recs = {}
    relevant = {1: {3}}
    result = evaluate(recs, relevant, k=3)
    assert result["recall_at_k"] == 0.0
