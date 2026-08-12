"""Metrics that Recall@K/NDCG@K don't capture: whether a model actually uses
the catalog, or just reliably recommends the same blockbusters to everyone.
A model can win on accuracy purely by suggesting popular movies (they're
popular because lots of people like them) while being useless as a
discovery tool -- these two numbers catch that.
"""


def catalog_coverage(recs: dict, candidate_universe: set) -> float:
    """Fraction of the candidate universe that shows up in *any* user's
    recommendation list. Low coverage means the model has a small effective
    vocabulary regardless of how big the catalog is.
    """
    if not candidate_universe:
        return float("nan")
    recommended_items = set()
    for items in recs.values():
        recommended_items.update(items)
    return round(len(recommended_items) / len(candidate_universe), 4)


def popularity_bias(recs: dict, top_decile_items: set) -> float:
    """Average fraction of each user's recommendation list that falls in the
    top 10% most-rated movies. High values mean the model is mostly reciting
    popularity, not personalizing.
    """
    fracs = []
    for items in recs.values():
        if not items:
            continue
        hits = sum(1 for i in items if i in top_decile_items)
        fracs.append(hits / len(items))
    return round(sum(fracs) / len(fracs), 4) if fracs else float("nan")


def top_decile_items(item_counts) -> set:
    """item_counts: a pandas Series of movieId -> train rating count."""
    n = max(1, len(item_counts) // 10)
    return set(item_counts.sort_values(ascending=False).head(n).index)
