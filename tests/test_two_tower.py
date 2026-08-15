import pandas as pd
import torch

from src.models.two_tower import build_vocab, id_dropout


def test_build_vocab_reserves_zero_for_unk():
    train = pd.DataFrame({"userId": [5, 5, 2], "movieId": [10, 20, 10]})
    user_ids, item_ids, user_to_idx, item_to_idx = build_vocab(train)

    # index 0 is reserved as UNK, so no real entity maps to it.
    assert 0 not in user_to_idx.values()
    assert 0 not in item_to_idx.values()
    assert sorted(user_to_idx.values()) == list(range(1, len(user_ids) + 1))
    assert sorted(item_to_idx.values()) == list(range(1, len(item_ids) + 1))


def test_id_dropout_zero_probability_is_identity():
    idx = torch.tensor([1, 2, 3, 4, 5])
    assert torch.equal(id_dropout(idx, 0.0), idx)


def test_id_dropout_full_probability_zeros_everything():
    idx = torch.tensor([1, 2, 3, 4, 5])
    result = id_dropout(idx, 1.0)
    assert torch.equal(result, torch.zeros_like(idx))


def test_id_dropout_does_not_mutate_input():
    idx = torch.tensor([1, 2, 3, 4, 5])
    id_dropout(idx, 1.0)
    assert torch.equal(idx, torch.tensor([1, 2, 3, 4, 5]))
