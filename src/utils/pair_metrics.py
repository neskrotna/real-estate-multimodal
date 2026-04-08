from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def summarize_pair_scores(labels: List[int], scores: List[float]) -> Dict[str, float]:
    if not labels:
        return {
            "num_pairs": 0,
            "roc_auc": 0.0,
            "average_precision": 0.0,
            "positive_mean_score": 0.0,
            "negative_mean_score": 0.0,
        }

    labels_np = np.array(labels)
    scores_np = np.array(scores)

    pos_scores = scores_np[labels_np == 1]
    neg_scores = scores_np[labels_np == 0]

    roc_auc = roc_auc_score(labels_np, scores_np) if len(set(labels)) > 1 else 0.0
    ap = average_precision_score(labels_np, scores_np) if len(set(labels)) > 1 else 0.0

    return {
        "num_pairs": int(len(labels)),
        "roc_auc": float(roc_auc),
        "average_precision": float(ap),
        "positive_mean_score": float(pos_scores.mean()) if len(pos_scores) else 0.0,
        "negative_mean_score": float(neg_scores.mean()) if len(neg_scores) else 0.0,
    }


def summarize_group_ranking(rows: List[dict], ks: List[int] = [1, 3, 5]) -> Dict[str, float]:
    """
    rows: list of dicts with keys:
        - text_listing_id
        - image_listing_id
        - label
        - score
    grouped by text_listing_id, rank candidates by score descending
    """
    groups = defaultdict(list)
    for row in rows:
        groups[row["text_listing_id"]].append(row)

    n = 0
    recalls = {k: 0.0 for k in ks}
    reciprocal_ranks = []

    for text_listing_id, candidates in groups.items():
        candidates_sorted = sorted(candidates, key=lambda x: x["score"], reverse=True)

        positive_positions = [
            idx + 1 for idx, c in enumerate(candidates_sorted) if int(c["label"]) == 1
        ]

        if not positive_positions:
            continue

        n += 1
        best_pos_rank = min(positive_positions)

        for k in ks:
            if best_pos_rank <= k:
                recalls[k] += 1.0

        reciprocal_ranks.append(1.0 / best_pos_rank)

    if n == 0:
        out = {f"recall@{k}": 0.0 for k in ks}
        out["mrr"] = 0.0
        out["num_queries"] = 0
        return out

    out = {f"recall@{k}": recalls[k] / n for k in ks}
    out["mrr"] = float(sum(reciprocal_ranks) / n)
    out["num_queries"] = int(n)
    return out