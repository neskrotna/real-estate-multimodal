from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict

import numpy as np
from sentence_transformers import SentenceTransformer

from src.utils.io import read_json, read_jsonl, write_json, write_jsonl
from src.utils.metrics import summarize_retrieval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SBERT text-only baseline: title -> description retrieval"
    )
    parser.add_argument("--listings", type=str, default="data/processed/listings.jsonl")
    parser.add_argument("--split", type=str, default="data/processed/split_v1.json")
    parser.add_argument("--run-dir", type=str, default="runs/text_sbert_title_to_description")
    parser.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return " ".join(str(text).split())


def load_testable_records(listings_path: str, split_path: str) -> Dict[str, List[dict]]:
    rows = read_jsonl(listings_path)
    split_map = read_json(split_path)

    train_ids = set(split_map.get("train", []))
    val_ids = set(split_map.get("val", []))
    test_ids = set(split_map.get("test", []))

    usable = []
    for row in rows:
        listing_id = str(row.get("listing_id", "")).strip()
        title = normalize_text(row.get("title", ""))
        description = normalize_text(row.get("description", ""))

        if not listing_id or not title or not description:
            continue

        usable.append(
            {
                "listing_id": listing_id,
                "title": title,
                "description": description,
            }
        )

    return {
        "train": [r for r in usable if r["listing_id"] in train_ids],
        "val": [r for r in usable if r["listing_id"] in val_ids],
        "test": [r for r in usable if r["listing_id"] in test_ids],
    }


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
    b_norm = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-12, None)
    return a_norm @ b_norm.T


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    data = load_testable_records(args.listings, args.split)

    if len(data["test"]) == 0:
        raise ValueError("No usable test records found with non-empty title and description.")

    test_titles = [r["title"] for r in data["test"]]
    test_descriptions = [r["description"] for r in data["test"]]
    test_ids = [r["listing_id"] for r in data["test"]]

    model = SentenceTransformer(args.model_name)

    query_emb = model.encode(
        test_titles,
        batch_size=args.batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    candidate_emb = model.encode(
        test_descriptions,
        batch_size=args.batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    similarity = cosine_similarity_matrix(query_emb, candidate_emb)

    results = []
    top5_rows = []

    for i, true_id in enumerate(test_ids):
        row_scores = similarity[i]
        ranked_idx = np.argsort(-row_scores)
        ranked_ids = [test_ids[j] for j in ranked_idx]

        results.append((true_id, ranked_ids))

        top5_rows.append(
            {
                "query_id": true_id,
                "query_title": test_titles[i],
                "predictions": [
                    {
                        "rank": rank + 1,
                        "candidate_id": test_ids[j],
                        "score": float(row_scores[j]),
                    }
                    for rank, j in enumerate(ranked_idx[:5])
                ],
            }
        )

    metrics = summarize_retrieval(results, ks=[1, 5, 10])

    write_json(run_dir / "metrics.json", metrics)
    write_jsonl(run_dir / "top5_predictions.jsonl", top5_rows)

    print("SBERT title->description baseline finished.")
    print(metrics)


if __name__ == "__main__":
    main()