from __future__ import annotations

import argparse
import json
from typing import Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.io import read_table, write_json
from src.utils.metrics import summarize_retrieval


def load_splits(path: str) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=str)
    parser.add_argument("--splits", required=True, type=str)
    parser.add_argument("--out", required=True, type=str)
    parser.add_argument("--k", nargs="+", type=int, default=[1, 5, 10])
    args = parser.parse_args()

    df = read_table(args.manifest)
    df["listing_id"] = df["listing_id"].astype(str)

    listing_text = df.groupby("listing_id")["text"].first().reset_index()

    splits = load_splits(args.splits)
    train_ids = set(map(str, splits["train"]))
    test_ids = list(map(str, splits["test"]))

    train_df = listing_text[listing_text["listing_id"].isin(train_ids)].copy()
    test_df = listing_text[listing_text["listing_id"].isin(test_ids)].copy()

    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_df["text"].fillna(""))
    X_test = vectorizer.transform(test_df["text"].fillna(""))

    sim = cosine_similarity(X_test, X_train)
    train_listing_ids = train_df["listing_id"].tolist()

    results = []
    for i, true_id in enumerate(test_df["listing_id"].tolist()):
        ranked_idx = np.argsort(-sim[i])
        ranked_ids = [train_listing_ids[j] for j in ranked_idx]
        results.append((true_id, ranked_ids))

    summary = summarize_retrieval(results, ks=args.k)

    out = {
        "baseline": "tfidf_text_retrieval",
        "split": "test_vs_train",
        "k": args.k,
        "metrics": summary,
        "n_train": len(train_df),
        "n_test": len(test_df),
    }

    write_json(args.out, out)
    print(f"Wrote report: {args.out}")
    print(out["metrics"])


if __name__ == "__main__":
    main()