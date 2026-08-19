import numpy as np
import pandas as pd
import torch

from src.models.two_tower import (
    ItemTower,
    TwoTowerRecommender,
    UserTower,
    build_vocab,
    id_dropout,
    load_checkpoint,
)


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


def test_load_checkpoint_reconstructs_towers_from_saved_state(tmp_path):
    torch.manual_seed(0)
    user_ids = np.array([10, 20, 30])
    item_ids = np.array([100, 200])
    genre_dim, genome_dim = 3, 2
    user_tower = UserTower(len(user_ids))
    item_tower = ItemTower(len(item_ids), genre_dim, genome_dim)

    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "user_tower": user_tower.state_dict(),
            "item_tower": item_tower.state_dict(),
            "user_ids": user_ids,
            "item_ids": item_ids,
        },
        checkpoint_path,
    )

    loaded_user_tower, loaded_item_tower, user_to_idx, item_to_idx, features = load_checkpoint(
        checkpoint_path, features={"genre_dim": genre_dim, "genome_dim": genome_dim}
    )

    assert user_to_idx == {10: 1, 20: 2, 30: 3}
    assert item_to_idx == {100: 1, 200: 2}
    for expected, actual in zip(user_tower.state_dict().values(), loaded_user_tower.state_dict().values()):
        assert torch.equal(expected, actual)
    for expected, actual in zip(item_tower.state_dict().values(), loaded_item_tower.state_dict().values()):
        assert torch.equal(expected, actual)


def test_recommend_batch_and_recommend_batch_with_scores_agree_on_order():
    torch.manual_seed(0)
    train = pd.DataFrame({"userId": [1, 1, 2], "movieId": [10, 20, 10]})
    user_ids, item_ids, user_to_idx, item_to_idx = build_vocab(train)

    catalog_movie_ids = np.array([10, 20, 30])
    genre_dim, genome_dim = 2, 2
    features = {
        "catalog_movie_ids": catalog_movie_ids,
        "catalog_movie_to_idx": {m: i for i, m in enumerate(catalog_movie_ids)},
        "genre_matrix": torch.rand(len(catalog_movie_ids), genre_dim),
        "genome_matrix": torch.rand(len(catalog_movie_ids), genome_dim),
        "has_genome": torch.ones(len(catalog_movie_ids)),
        "genre_dim": genre_dim,
        "genome_dim": genome_dim,
    }

    user_tower = UserTower(len(user_ids))
    item_tower = ItemTower(len(item_ids), genre_dim, genome_dim)
    model = TwoTowerRecommender(user_tower, item_tower, user_to_idx, item_to_idx, features, train)

    ids_only = model.recommend_batch([1, 2], k=2)
    with_scores = model.recommend_batch_with_scores([1, 2], k=2)

    for user_id in [1, 2]:
        assert ids_only[user_id] == [movie_id for movie_id, _ in with_scores[user_id]]
