from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

def recall_at_k(true_id: str, ranked_ids: Sequence[str], k: int) -> float:
    return 1.0 if true_id in ranked_ids[:k] else 0.0

def mrr(true_id: str, ranked_ids: Sequence[str]) -> float:
    for idx, rid in enumerate(ranked_ids, start=1):
        if rid == true_id:
            return 1.0 / idx
    return 0.0

def summarize_retrieval(results: List[Tuple[str, List[str]]], ks: List[int]) -> Dict[str, float]:
    """
    results: list of (true_listing_id, ranked_listing_ids)
    """
    n = len(results)
    if n == 0:
        out = {f"recall@{k}": 0.0 for k in ks}
        out["mrr"] = 0.0
        return out

    out: Dict[str, float] = {}
    for k in ks:
        out[f"recall@{k}"] = sum(recall_at_k(t, r, k) for t, r in results) / n
    out["mrr"] = sum(mrr(t, r) for t, r in results) / n
    return out