"""Compares all models trained so far on the same candidate universe and
val users, and separately benchmarks brute-force vs. FAISS retrieval
latency -- first at the real catalog size (~14K movies, where the two-stage
retrieval pattern isn't needed yet) and then at synthetic catalog sizes
(100K / 1M items) that approximate a real production catalog, to show
*where* approximate nearest-neighbor search actually starts to matter. Both
model accuracy and latency findings feed the README's "what worked / what
didn't" section directly.

Usage:
    python -m src.models.compare
"""

import json
import time

import faiss
import numpy as np
import pandas as pd

from src.data.config import PROCESSED_DIR, RANDOM_SEED, ROOT_DIR
from src.eval.diversity import catalog_coverage, popularity_bias, top_decile_items
from src.eval.metrics import evaluate
from src.models.baseline import ItemItemCF
from src.models.baseline import MIN_ITEM_TRAIN_RATINGS as CF_MIN_RATINGS
from src.models.matrix_factorization import MatrixFactorization
from src.models.matrix_factorization import N_FACTORS
from src.models.matrix_factorization import MIN_ITEM_TRAIN_RATINGS as MF_MIN_RATINGS

TOP_K = 10
LATENCY_QUERIES_REAL = 500
LATENCY_QUERIES_SYNTHETIC = 100
SYNTHETIC_CATALOG_SIZES = [100_000, 1_000_000]


def eval_model(name, recs, relevant, item_universe, item_counts, decile_items):
    metrics = evaluate(recs, relevant, TOP_K)
    return {
        "model": name,
        **metrics,
        "catalog_coverage": catalog_coverage(recs, item_universe),
        "popularity_bias_top_decile_share": popularity_bias(recs, decile_items),
    }


def benchmark_retrieval(item_vectors: np.ndarray, query_vectors: np.ndarray, k: int) -> dict:
    """Brute-force (numpy) vs FAISS exact (IndexFlatIP) vs FAISS approximate
    (IndexIVFFlat) top-k retrieval latency, at whatever catalog size
    item_vectors happens to be. Also reports how much overlap the
    approximate index has with the exact result, since a latency win that
    silently drops relevant items isn't free.
    """
    n_items, dim = item_vectors.shape
    item_vectors = np.ascontiguousarray(item_vectors, dtype=np.float32)
    query_vectors = np.ascontiguousarray(query_vectors, dtype=np.float32)

    # Brute force: plain numpy matmul + argpartition, one query at a time
    # (mirrors how a single request would be served, not a batched job).
    start = time.perf_counter()
    brute_results = []
    for q in query_vectors:
        scores = item_vectors @ q
        top_idx = np.argpartition(-scores, k)[:k]
        brute_results.append(set(top_idx[np.argsort(-scores[top_idx])]))
    brute_ms = (time.perf_counter() - start) / len(query_vectors) * 1000

    # FAISS exact search -- same results as brute force, different implementation.
    flat_index = faiss.IndexFlatIP(dim)
    flat_index.add(item_vectors)
    start = time.perf_counter()
    _, flat_idx = flat_index.search(query_vectors, k)
    flat_ms = (time.perf_counter() - start) / len(query_vectors) * 1000
    flat_results = [set(row) for row in flat_idx]

    result = {
        "n_items": n_items,
        "dim": dim,
        "numpy_brute_force_ms_per_query": round(brute_ms, 4),
        "faiss_flat_exact_ms_per_query": round(flat_ms, 4),
    }

    # Approximate search only makes sense to benchmark once there's enough
    # data to build meaningful clusters; skip it for tiny catalogs.
    if n_items >= 10_000:
        nlist = max(1, int(np.sqrt(n_items)))
        quantizer = faiss.IndexFlatIP(dim)
        ivf_index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        ivf_index.train(item_vectors)
        ivf_index.add(item_vectors)
        ivf_index.nprobe = 8
        start = time.perf_counter()
        _, ivf_idx = ivf_index.search(query_vectors, k)
        ivf_ms = (time.perf_counter() - start) / len(query_vectors) * 1000
        ivf_results = [set(row) for row in ivf_idx]

        overlaps = [
            len(a & b) / k for a, b in zip(flat_results, ivf_results) if len(a) and len(b)
        ]
        result["faiss_ivf_approx_ms_per_query"] = round(ivf_ms, 4)
        result["faiss_ivf_nlist"] = nlist
        result["faiss_ivf_nprobe"] = 8
        result["faiss_ivf_overlap_with_exact_at_k"] = round(sum(overlaps) / len(overlaps), 4)

    return result


def main() -> None:
    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    val = pd.read_parquet(PROCESSED_DIR / "val.parquet")

    assert CF_MIN_RATINGS == MF_MIN_RATINGS, "models must share a candidate universe to compare fairly"

    print("Fitting item-item CF...")
    cf = ItemItemCF(min_item_train_ratings=CF_MIN_RATINGS).fit(train)
    print("Fitting matrix factorization...")
    mf = MatrixFactorization(min_item_train_ratings=MF_MIN_RATINGS).fit(train)

    item_universe = set(cf.item_ids_)
    val_users = set(val["userId"].unique())
    warm_val_users = val_users & set(cf.user_ids_)
    relevant = {}
    for user_id, group in val[val["userId"].isin(warm_val_users)].groupby("userId"):
        relevant[user_id] = set(group["movieId"]) & item_universe

    item_counts = train[train["movieId"].isin(item_universe)].groupby("movieId").size()
    decile_items = top_decile_items(item_counts)
    top10_pop = list(item_counts.sort_values(ascending=False).head(TOP_K).index)

    user_ids = list(relevant.keys())
    print(f"Generating recommendations for {len(user_ids):,} warm val users (3 models)...")
    pop_recs = {u: top10_pop for u in user_ids}
    cf_recs = cf.recommend_batch(user_ids, TOP_K)
    mf_recs = mf.recommend_batch(user_ids, TOP_K)

    model_results = [
        eval_model("popularity", pop_recs, relevant, item_universe, item_counts, decile_items),
        eval_model("item_item_cf", cf_recs, relevant, item_universe, item_counts, decile_items),
        eval_model("matrix_factorization", mf_recs, relevant, item_universe, item_counts, decile_items),
    ]
    print("\nModel comparison:")
    print(json.dumps(model_results, indent=2))

    # --- Retrieval latency: real catalog scale ---
    print(f"\nBenchmarking retrieval latency at real catalog scale ({len(mf.item_ids_):,} items)...")
    rng = np.random.default_rng(RANDOM_SEED)
    query_idx = rng.choice(len(mf.user_factors_), size=LATENCY_QUERIES_REAL, replace=False)
    real_scale = benchmark_retrieval(mf.item_factors_, mf.user_factors_[query_idx], TOP_K)

    # --- Retrieval latency: synthetic production-scale catalogs ---
    # The real catalog (~14K items) is too small for brute force to ever be a
    # bottleneck. These sizes approximate what a real streaming catalog looks
    # like, using synthetic vectors matched to the real embeddings' mean/std,
    # to show where the two-stage retrieval pattern actually earns its keep.
    synthetic_scale = []
    item_mean, item_std = mf.item_factors_.mean(), mf.item_factors_.std()
    for n_items in SYNTHETIC_CATALOG_SIZES:
        print(f"Benchmarking retrieval latency at synthetic scale ({n_items:,} items)...")
        synth_items = rng.normal(item_mean, item_std, size=(n_items, N_FACTORS)).astype(np.float32)
        query_sample_idx = rng.choice(
            len(mf.user_factors_), size=LATENCY_QUERIES_SYNTHETIC, replace=False
        )
        bench = benchmark_retrieval(synth_items, mf.user_factors_[query_sample_idx], TOP_K)
        synthetic_scale.append(bench)

    latency_results = {"real_scale": real_scale, "synthetic_scale": synthetic_scale}
    print("\nRetrieval latency:")
    print(json.dumps(latency_results, indent=2))

    with open(ROOT_DIR / "docs" / "model_comparison.json", "w") as f:
        json.dump({"models": model_results, "latency": latency_results}, f, indent=2)


if __name__ == "__main__":
    main()
