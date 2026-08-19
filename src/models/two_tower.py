"""Two-tower retrieval model: a user tower (ID embedding only -- MovieLens
has no user side-info, so this is a real, stated limit on fixing cold-start
*users*) and an item tower (ID embedding + genre + tag-genome content
features), trained with in-batch sampled softmax and a log-Q popularity
correction so the model isn't just rewarded for recommending whatever's
already popular (the concrete failure mode the baseline/MF comparison
found).

Cold-start handling is trained, not just structurally possible: ID index 0
is reserved as UNK in both towers, and during training each example's ID is
randomly zeroed out to UNK with some probability so the UNK row actually
receives gradient and the item tower is forced to sometimes rely on content
features alone. Without this, the UNK row would never be updated and would
be useless the one time it matters -- scoring a movie added after training.

Usage:
    python -m src.models.two_tower
"""

import json
import time

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.config import PROCESSED_DIR, RANDOM_SEED, ROOT_DIR
from src.eval.diversity import catalog_coverage, popularity_bias, top_decile_items
from src.eval.metrics import evaluate

ID_DIM = 32
CONTENT_HIDDEN = 128
OUT_DIM = 32
BATCH_SIZE = 4096
EPOCHS = 2
LR = 5e-3
WEIGHT_DECAY = 1e-6
ID_DROPOUT_PROB = 0.15
TOP_K = 10
CHECKPOINT_PATH = PROCESSED_DIR / "two_tower_checkpoint.pt"

torch.manual_seed(RANDOM_SEED)


# --------------------------------------------------------------------------
# Content features, aligned to one stable "catalog index" covering every
# movie that appears anywhere in train/val/test -- not just train, since the
# whole point is to be able to score movies train never saw.
# --------------------------------------------------------------------------
def load_catalog_features():
    genre_df = pd.read_parquet(PROCESSED_DIR / "item_genre_features.parquet")
    genome_ids = np.load(PROCESSED_DIR / "item_genome_movie_ids.npy")
    genome_matrix = np.load(PROCESSED_DIR / "item_genome_features.npy")

    catalog_movie_ids = np.sort(genre_df["movieId"].unique())
    catalog_movie_to_idx = {m: i for i, m in enumerate(catalog_movie_ids)}

    genre_cols = [c for c in genre_df.columns if c != "movieId"]
    genre_df = genre_df.set_index("movieId").loc[catalog_movie_ids]
    genre_matrix = np.ascontiguousarray(genre_df[genre_cols].to_numpy(dtype=np.float32))

    genome_dim = genome_matrix.shape[1]
    full_genome = np.zeros((len(catalog_movie_ids), genome_dim), dtype=np.float32)
    has_genome = np.zeros(len(catalog_movie_ids), dtype=np.float32)
    genome_row_by_movie = {m: i for i, m in enumerate(genome_ids)}
    for movie_id, catalog_idx in catalog_movie_to_idx.items():
        row = genome_row_by_movie.get(movie_id)
        if row is not None:
            full_genome[catalog_idx] = genome_matrix[row]
            has_genome[catalog_idx] = 1.0

    return {
        "catalog_movie_ids": catalog_movie_ids,
        "catalog_movie_to_idx": catalog_movie_to_idx,
        "genre_matrix": torch.from_numpy(genre_matrix),
        "genome_matrix": torch.from_numpy(full_genome),
        "has_genome": torch.from_numpy(has_genome),
        "genre_dim": genre_matrix.shape[1],
        "genome_dim": genome_dim,
    }


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
class UserTower(nn.Module):
    def __init__(self, n_train_users: int, id_dim: int = ID_DIM, out_dim: int = OUT_DIM):
        super().__init__()
        self.id_embedding = nn.Embedding(n_train_users + 1, id_dim, padding_idx=0)
        self.mlp = nn.Sequential(nn.Linear(id_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim))

    def forward(self, user_id_idx: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.id_embedding(user_id_idx))


class ItemTower(nn.Module):
    def __init__(
        self,
        n_train_items: int,
        genre_dim: int,
        genome_dim: int,
        id_dim: int = ID_DIM,
        content_hidden: int = CONTENT_HIDDEN,
        out_dim: int = OUT_DIM,
    ):
        super().__init__()
        self.id_embedding = nn.Embedding(n_train_items + 1, id_dim, padding_idx=0)
        self.content_mlp = nn.Sequential(
            nn.Linear(genre_dim + genome_dim + 1, content_hidden),
            nn.ReLU(),
            nn.Linear(content_hidden, id_dim),
        )
        self.output_mlp = nn.Sequential(
            nn.Linear(id_dim * 2, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)
        )

    def forward(
        self,
        item_id_idx: torch.Tensor,
        genre: torch.Tensor,
        genome: torch.Tensor,
        has_genome: torch.Tensor,
    ) -> torch.Tensor:
        id_vec = self.id_embedding(item_id_idx)
        content_input = torch.cat([genre, genome, has_genome.unsqueeze(-1)], dim=-1)
        content_vec = self.content_mlp(content_input)
        return self.output_mlp(torch.cat([id_vec, content_vec], dim=-1))


def id_dropout(idx: torch.Tensor, prob: float) -> torch.Tensor:
    if prob <= 0:
        return idx
    mask = torch.rand(idx.shape) < prob
    return idx.masked_fill(mask, 0)


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def build_vocab(train: pd.DataFrame):
    user_ids = np.sort(train["userId"].unique())
    item_ids = np.sort(train["movieId"].unique())
    # index 0 reserved for UNK in both towers; real train entities start at 1.
    user_to_idx = {u: i + 1 for i, u in enumerate(user_ids)}
    item_to_idx = {m: i + 1 for i, m in enumerate(item_ids)}
    return user_ids, item_ids, user_to_idx, item_to_idx


def train_two_tower(train: pd.DataFrame, features: dict):
    user_ids, item_ids, user_to_idx, item_to_idx = build_vocab(train)
    catalog_movie_to_idx = features["catalog_movie_to_idx"]

    user_idx_arr = train["userId"].map(user_to_idx).to_numpy(dtype=np.int64)
    item_id_idx_arr = train["movieId"].map(item_to_idx).to_numpy(dtype=np.int64)
    item_catalog_idx_arr = train["movieId"].map(catalog_movie_to_idx).to_numpy(dtype=np.int64)

    # log-Q popularity correction: exact offline train frequency per catalog item
    # (the original paper streams this estimate online for unbounded data; here
    # the full training set is static, so the exact count is simpler and exact).
    item_freq = np.zeros(len(features["catalog_movie_ids"]), dtype=np.float64)
    counts = pd.Series(item_catalog_idx_arr).value_counts()
    item_freq[counts.index.to_numpy()] = counts.to_numpy()
    item_prob = item_freq / item_freq.sum()
    log_q = torch.from_numpy(np.log(item_prob + 1e-12)).float()

    dataset = TensorDataset(
        torch.from_numpy(user_idx_arr.copy()),
        torch.from_numpy(item_id_idx_arr.copy()),
        torch.from_numpy(item_catalog_idx_arr.copy()),
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    user_tower = UserTower(len(user_ids))
    item_tower = ItemTower(len(item_ids), features["genre_dim"], features["genome_dim"])
    optimizer = torch.optim.Adam(
        list(user_tower.parameters()) + list(item_tower.parameters()), lr=LR, weight_decay=WEIGHT_DECAY
    )
    loss_fn = nn.CrossEntropyLoss()

    genre_matrix, genome_matrix, has_genome = (
        features["genre_matrix"],
        features["genome_matrix"],
        features["has_genome"],
    )

    print(
        f"Training two-tower model: {len(user_ids):,} users, {len(item_ids):,} train items, "
        f"{len(loader):,} batches/epoch, {EPOCHS} epochs"
    )
    step = 0
    start = time.time()
    for epoch in range(EPOCHS):
        running_loss, running_n = 0.0, 0
        for user_batch, item_id_batch, item_cat_batch in loader:
            optimizer.zero_grad()

            user_vecs = user_tower(id_dropout(user_batch, ID_DROPOUT_PROB))
            item_vecs = item_tower(
                id_dropout(item_id_batch, ID_DROPOUT_PROB),
                genre_matrix[item_cat_batch],
                genome_matrix[item_cat_batch],
                has_genome[item_cat_batch],
            )

            logits = user_vecs @ item_vecs.T
            logits = logits - log_q[item_cat_batch].unsqueeze(0)
            target = torch.arange(logits.shape[0])
            loss = loss_fn(logits, target)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(user_batch)
            running_n += len(user_batch)
            step += 1
            if step % 500 == 0:
                elapsed = time.time() - start
                print(
                    f"  epoch {epoch + 1}/{EPOCHS} step {step:,} "
                    f"loss={running_loss / running_n:.4f} elapsed={elapsed:.0f}s"
                )
                running_loss, running_n = 0.0, 0

        print(f"epoch {epoch + 1}/{EPOCHS} done, elapsed={time.time() - start:.0f}s")

    torch.save(
        {
            "user_tower": user_tower.state_dict(),
            "item_tower": item_tower.state_dict(),
            "user_ids": user_ids,
            "item_ids": item_ids,
        },
        CHECKPOINT_PATH,
    )
    print(f"Saved checkpoint to {CHECKPOINT_PATH}")
    return user_tower, item_tower, user_to_idx, item_to_idx


def load_checkpoint(checkpoint_path=CHECKPOINT_PATH, features: dict | None = None):
    """Inverse of the torch.save() at the end of train_two_tower(): rebuilds
    both towers from a saved checkpoint without retraining. Nothing else in
    the codebase reads this checkpoint back -- ranking/serving code needs a
    trained model without paying the ~27 min training cost again.
    """
    # weights_only=False: the checkpoint bundles user_ids/item_ids as numpy
    # arrays alongside the tensors, and we only ever load checkpoints this
    # codebase wrote itself, so there's no untrusted-source risk here.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    user_ids, item_ids = checkpoint["user_ids"], checkpoint["item_ids"]
    # same convention as build_vocab(): index 0 reserved for UNK.
    user_to_idx = {u: i + 1 for i, u in enumerate(user_ids)}
    item_to_idx = {m: i + 1 for i, m in enumerate(item_ids)}

    if features is None:
        features = load_catalog_features()

    user_tower = UserTower(len(user_ids))
    item_tower = ItemTower(len(item_ids), features["genre_dim"], features["genome_dim"])
    user_tower.load_state_dict(checkpoint["user_tower"])
    item_tower.load_state_dict(checkpoint["item_tower"])
    user_tower.eval()
    item_tower.eval()
    return user_tower, item_tower, user_to_idx, item_to_idx, features


def load_user_tower(checkpoint_path=CHECKPOINT_PATH):
    """Loads just the user tower + user_to_idx from a checkpoint, skipping
    the item tower entirely. Serving only needs to embed a user at request
    time -- item embeddings are precomputed into a FAISS index ahead of time
    (src/serving/build_index.py) -- so this avoids reconstructing ItemTower
    and, with it, ever touching the 59MB genome feature files at serving
    time; only the small checkpoint file is read.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    user_ids = checkpoint["user_ids"]
    user_to_idx = {u: i + 1 for i, u in enumerate(user_ids)}
    user_tower = UserTower(len(user_ids))
    user_tower.load_state_dict(checkpoint["user_tower"])
    user_tower.eval()
    return user_tower, user_to_idx


# --------------------------------------------------------------------------
# Inference: precompute embeddings for every train user and every catalog
# item (using each item's real ID if seen in train, UNK otherwise), then
# recommend via brute-force dot product excluding already-seen items --
# mirrors the baseline/MF recommend_batch interface for reuse with the same
# eval utilities.
# --------------------------------------------------------------------------
class TwoTowerRecommender:
    def __init__(self, user_tower, item_tower, user_to_idx, item_to_idx, features, train):
        self.catalog_movie_ids = features["catalog_movie_ids"]
        self.item_ids_ = self.catalog_movie_ids  # full catalog, not just train items
        self.user_ids_ = np.array(sorted(user_to_idx))
        self.user_to_idx_ = user_to_idx

        user_tower.eval()
        item_tower.eval()
        with torch.no_grad():
            all_user_idx = torch.arange(1, len(user_to_idx) + 1)
            self.user_embeddings_ = user_tower(all_user_idx).numpy()

            item_id_idx = torch.zeros(len(self.catalog_movie_ids), dtype=torch.long)
            for movie_id, i in item_to_idx.items():
                catalog_pos = features["catalog_movie_to_idx"][movie_id]
                item_id_idx[catalog_pos] = i
            self.item_embeddings_ = item_tower(
                item_id_idx, features["genre_matrix"], features["genome_matrix"], features["has_genome"]
            ).numpy()

        catalog_idx_of = features["catalog_movie_to_idx"]
        rows = train["userId"].map(self.user_to_idx_).to_numpy() - 1
        cols = train["movieId"].map(catalog_idx_of).to_numpy()
        self.X_ = sparse.csr_matrix(
            (np.ones(len(train), dtype=np.float32), (rows, cols)),
            shape=(len(self.user_ids_), len(self.catalog_movie_ids)),
        )

    def _top_k(self, user_ids: list, k: int, batch_size: int = 1000) -> dict:
        """Shared by recommend_batch (ids only) and recommend_batch_with_scores
        (ids + raw dot-product score, needed as a ranking-stage feature)."""
        known = [u for u in user_ids if u in self.user_to_idx_]
        recs = {}
        for start in range(0, len(known), batch_size):
            batch = known[start : start + batch_size]
            idx = [self.user_to_idx_[u] - 1 for u in batch]
            seen = self.X_[idx]
            scores = self.user_embeddings_[idx] @ self.item_embeddings_.T
            scores[seen.toarray() > 0] = -np.inf
            top_k_idx = np.argpartition(-scores, k, axis=1)[:, :k]
            for row, user_id in enumerate(batch):
                row_scores = scores[row, top_k_idx[row]]
                order = np.argsort(-row_scores)
                ranked_idx = top_k_idx[row][order]
                recs[user_id] = [
                    (self.catalog_movie_ids[i], float(scores[row, i]))
                    for i in ranked_idx
                    if scores[row, i] > -np.inf
                ]
        return recs

    def recommend_batch(self, user_ids: list, k: int, batch_size: int = 1000) -> dict:
        return {
            user_id: [movie_id for movie_id, _ in ranked]
            for user_id, ranked in self._top_k(user_ids, k, batch_size).items()
        }

    def recommend_batch_with_scores(self, user_ids: list, k: int, batch_size: int = 1000) -> dict:
        return self._top_k(user_ids, k, batch_size)


def main() -> None:
    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    val = pd.read_parquet(PROCESSED_DIR / "val.parquet")

    print("Loading content features...")
    features = load_catalog_features()
    print(
        f"  catalog: {len(features['catalog_movie_ids']):,} movies, "
        f"genre_dim={features['genre_dim']}, genome_dim={features['genome_dim']}"
    )

    user_tower, item_tower, user_to_idx, item_to_idx = train_two_tower(train, features)
    model = TwoTowerRecommender(user_tower, item_tower, user_to_idx, item_to_idx, features, train)

    # Full catalog this time -- no min-ratings filter -- since closing that
    # coverage gap is what the content features are for.
    item_universe = set(model.item_ids_)
    val_users = set(val["userId"].unique())
    warm_val_users = val_users & set(model.user_ids_)
    relevant = {}
    for user_id, group in val[val["userId"].isin(warm_val_users)].groupby("userId"):
        relevant[user_id] = set(group["movieId"]) & item_universe

    print(f"Generating top-{TOP_K} recommendations for {len(relevant):,} warm val users...")
    recs = model.recommend_batch(list(relevant.keys()), TOP_K)

    item_counts = train.groupby("movieId").size()
    decile_items = top_decile_items(item_counts)
    metrics = evaluate(recs, relevant, TOP_K)
    result = {
        "metrics": metrics,
        "catalog_coverage": catalog_coverage(recs, item_universe),
        "popularity_bias_top_decile_share": popularity_bias(recs, decile_items),
        "coverage": {
            "val_users_total": len(val_users),
            "val_users_warm_pct": round(100 * len(warm_val_users) / len(val_users), 1),
            "candidate_movies": len(item_universe),
        },
    }
    print(json.dumps(result, indent=2))

    # Cold-start-only slice: movies never seen in train. Baseline/MF structurally
    # cannot recommend these at all (0% coverage by construction); this is the
    # test of whether the content features actually do anything.
    train_movies = set(train["movieId"])
    cold_relevant = {u: (items - train_movies) for u, items in relevant.items()}
    cold_relevant = {u: items for u, items in cold_relevant.items() if items}
    cold_metrics = evaluate(recs, cold_relevant, TOP_K)
    print("\nCold-start-only slice (val movies never seen in train):")
    print(json.dumps(cold_metrics, indent=2))
    result["cold_start_only"] = cold_metrics

    with open(ROOT_DIR / "docs" / "two_tower_eval.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
