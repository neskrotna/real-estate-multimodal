from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.utils.io import read_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 1: multilingual CLIP with max-image similarity pooling."
    )
    parser.add_argument("--val-pairs", type=Path, default=Path("data/processed/pairs_val.jsonl"))
    parser.add_argument("--test-pairs", type=Path, default=Path("data/processed/pairs_test.jsonl"))
    parser.add_argument(
        "--text-model",
        type=str,
        default="sentence-transformers/clip-ViT-B-32-multilingual-v1",
    )
    parser.add_argument(
        "--image-model",
        type=str,
        default="clip-ViT-B-32",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs/exp1_clip_maxpool"))
    parser.add_argument("--max-images-per-pair", type=int, default=4)
    return parser.parse_args()


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def f1_at_threshold(scored_pairs: list[dict[str, Any]], threshold: float) -> float:
    tp = fp = fn = 0
    for row in scored_pairs:
        pred = 1 if row["similarity"] >= threshold else 0
        label = row["label"]
        if pred == 1 and label == 1:
            tp += 1
        elif pred == 1 and label == 0:
            fp += 1
        elif pred == 0 and label == 1:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)


def compute_metrics(scored_pairs: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    tp = fp = tn = fn = 0

    for row in scored_pairs:
        pred = 1 if row["similarity"] >= threshold else 0
        label = row["label"]

        if pred == 1 and label == 1:
            tp += 1
        elif pred == 1 and label == 0:
            fp += 1
        elif pred == 0 and label == 0:
            tn += 1
        elif pred == 0 and label == 1:
            fn += 1

    total = len(scored_pairs)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)

    return {
        "num_pairs": total,
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def find_best_threshold(scored_val_pairs: list[dict[str, Any]]) -> tuple[float, float]:
    best_threshold = 0.0
    best_f1 = -1.0

    for i in range(-100, 101):
        threshold = i / 100.0
        f1 = f1_at_threshold(scored_val_pairs, threshold)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold, best_f1


def main() -> None:
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Loading multilingual text model: {args.text_model}")
    print(f"[INFO] Loading image model: {args.image_model}")

    text_model = SentenceTransformer(args.text_model, device=device)
    image_model = SentenceTransformer(args.image_model, device=device)

    val_pairs = read_jsonl(args.val_pairs)
    test_pairs = read_jsonl(args.test_pairs)

    text_cache: dict[str, np.ndarray] = {}
    image_cache: dict[str, tuple[list[str], np.ndarray]] = {}

    def get_text_embedding(pair: dict[str, Any]) -> np.ndarray:
        listing_id = pair["text_listing_id"]
        if listing_id not in text_cache:
            emb = text_model.encode(
                [pair["text"]],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )[0]
            text_cache[listing_id] = emb
        return text_cache[listing_id]

    def get_image_embeddings(pair: dict[str, Any]) -> tuple[list[str], np.ndarray]:
        listing_id = pair["image_listing_id"]
        if listing_id not in image_cache:
            selected_paths = pair["image_paths"][: args.max_images_per_pair]
            images = [load_image(p) for p in selected_paths]
            embs = image_model.encode(
                images,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            image_cache[listing_id] = (selected_paths, embs)
        return image_cache[listing_id]

    def score_pairs(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        scored: list[dict[str, Any]] = []

        for pair in tqdm(pairs, desc="Scoring pairs"):
            text_emb = get_text_embedding(pair)
            selected_paths, image_embs = get_image_embeddings(pair)

            sims = image_embs @ text_emb
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])

            row = dict(pair)
            row["similarity"] = best_sim
            row["best_image_path"] = selected_paths[best_idx]
            row["all_image_similarities"] = [float(x) for x in sims.tolist()]
            scored.append(row)

        return scored

    scored_val = score_pairs(val_pairs)
    best_threshold, best_val_f1 = find_best_threshold(scored_val)
    val_metrics = compute_metrics(scored_val, best_threshold)

    print(f"[INFO] Best validation threshold: {best_threshold:.2f}")
    print(f"[INFO] Best validation F1: {best_val_f1:.4f}")

    scored_test = score_pairs(test_pairs)
    test_metrics = compute_metrics(scored_test, best_threshold)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "scored_val.jsonl", scored_val)
    write_jsonl(args.output_dir / "scored_test.jsonl", scored_test)
    write_json(args.output_dir / "val_metrics.json", val_metrics)
    write_json(args.output_dir / "test_metrics.json", test_metrics)

    print("[INFO] Validation metrics:")
    print(json.dumps(val_metrics, indent=2, ensure_ascii=False))
    print("[INFO] Test metrics:")
    print(json.dumps(test_metrics, indent=2, ensure_ascii=False))
    print(f"[INFO] Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()