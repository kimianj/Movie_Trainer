import math


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    if not relevant:
        return math.nan
    hits = len(set(recommended[:k]) & relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    if not relevant:
        return math.nan
    dcg = sum(
        1.0 / math.log2(i + 2) for i, item in enumerate(recommended[:k]) if item in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate(recs: dict, relevant: dict, k: int) -> dict:
    """recs: user_id -> ranked list of item_ids. relevant: user_id -> set of held-out item_ids.

    Users with an empty relevant set are skipped (nothing to measure against),
    and the skipped count is reported rather than silently dropped.
    """
    recalls, ndcgs = [], []
    skipped = 0
    for user_id, rel in relevant.items():
        if not rel:
            skipped += 1
            continue
        ranked = recs.get(user_id, [])
        recalls.append(recall_at_k(ranked, rel, k))
        ndcgs.append(ndcg_at_k(ranked, rel, k))

    return {
        "k": k,
        "n_users_evaluated": len(recalls),
        "n_users_skipped_no_relevant_items": skipped,
        "recall_at_k": round(sum(recalls) / len(recalls), 4) if recalls else math.nan,
        "ndcg_at_k": round(sum(ndcgs) / len(ndcgs), 4) if ndcgs else math.nan,
    }
