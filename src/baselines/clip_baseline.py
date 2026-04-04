from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

from src.utils.io import read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multilingual CLIP baseline on German text-image pairs."
    )
    parser.add_argument(
        "--val-pairs",
        type=Path,
        default=Path("data/processed/pairs_val.jsonl"),
        help="Path to validation pairs JSONL",
    )
    parser.add_argument(
        "--test-pairs",
        type=Path,
        default=Path("data/processed/pairs_test.jsonl"),
        help="Path to test pairs JSONL",
    )
    parser.add_argument(
        "--text-model",
        type=str,
        default="sentence-transformers/clip-ViT-B-32-multilingual-v1",
        help="Multilingual text model",
    )
    parser.add_argument(
        "--image-model",
        type=str,
        default="clip-ViT-B-32",
        help="Image model aligned to the multilingual text model",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/german_clip_baseline"),
        help="Directory where results will be saved",
    )
    parser.add_argument(
        "--max-images-per-pair",
        type=int,
        default=4,
        help="Maximum number of images to use per pair",
    )
    return parser.parse_args()


def load_image(image_path: str) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def encode_text(
    text: str,
    text_model: SentenceTransformer,
) -> np.ndarray:
    emb = text_model.encode(
        [text],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return emb[0]


def encode_images(
    image_paths: list[str],
    image_model: SentenceTransformer,
    max_images_per_pair: int,
) -> np.ndarray:
    selected_paths = image_paths[:max_images_per_pair]

    if len(selected_paths) == 0:
        raise ValueError("Pair has no image paths.")

    images = [load_image(path) for path in selected_paths]

    embs = image_model.encode(
        images,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    pooled = np.mean(embs, axis=0)
    norm = np.linalg.norm(pooled)

    if norm == 0:
        return pooled

    return pooled / norm


def score_pair(
    pair: dict[str, Any],
    text_model: SentenceTransformer,
    image_model: SentenceTransformer,
    max_images_per_pair: int,
) -> float:
    text = pair["text"]
    image_paths = pair["image_paths"]

    text_emb = encode_text(text=text, text_model=text_model)
    image_emb = encode_images(
        image_paths=image_paths,
        image_model=image_model,
        max_images_per_pair=max_images_per_pair,
    )

    similarity = float(np.dot(text_emb, image_emb))
    return similarity


def score_pairs(
    pairs: list[dict[str, Any]],
    text_model: SentenceTransformer,
    image_model: SentenceTransformer,
    max_images_per_pair: int,
) -> list[dict[str, Any]]:
    scored_pairs: list[dict[str, Any]] = []

    for pair in tqdm(pairs, desc="Scoring pairs"):
        similarity = score_pair(
            pair=pair,
            text_model=text_model,
            image_model=image_model,
            max_images_per_pair=max_images_per_pair,
        )

        row = dict(pair)
        row["similarity"] = similarity
        scored_pairs.append(row)

    return scored_pairs


def accuracy_at_threshold(
    scored_pairs: list[dict[str, Any]],
    threshold: float,
) -> float:
    if not scored_pairs:
        return 0.0

    correct = 0
    for row in scored_pairs:
        pred = 1 if row["similarity"] >= threshold else 0
        if pred == row["label"]:
            correct += 1

    return correct / len(scored_pairs)


def find_best_threshold(
    scored_val_pairs: list[dict[str, Any]],
) -> tuple[float, float]:
    best_threshold = 0.0
    best_accuracy = -1.0

    for i in range(-100, 101):
        threshold = i / 100.0
        acc = accuracy_at_threshold(scored_val_pairs, threshold)

        if acc > best_accuracy:
            best_accuracy = acc
            best_threshold = threshold

    return best_threshold, best_accuracy


def compute_metrics(
    scored_pairs: list[dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    total = len(scored_pairs)
    correct = 0

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for row in scored_pairs:
        label = row["label"]
        pred = 1 if row["similarity"] >= threshold else 0

        if pred == label:
            correct += 1

        if label == 1 and pred == 1:
            tp += 1
        elif label == 0 and pred == 1:
            fp += 1
        elif label == 0 and pred == 0:
            tn += 1
        elif label == 1 and pred == 0:
            fn += 1

    accuracy = correct / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

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


def save_scored_jsonl(
    path: Path,
    scored_pairs: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in scored_pairs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


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

    print(f"[INFO] Loaded {len(val_pairs)} val pairs")
    print(f"[INFO] Loaded {len(test_pairs)} test pairs")

    scored_val = score_pairs(
        pairs=val_pairs,
        text_model=text_model,
        image_model=image_model,
        max_images_per_pair=args.max_images_per_pair,
    )

    best_threshold, best_val_accuracy = find_best_threshold(scored_val)
    val_metrics = compute_metrics(scored_val, best_threshold)

    print(f"[INFO] Best validation threshold: {best_threshold:.2f}")
    print(f"[INFO] Validation accuracy: {best_val_accuracy:.4f}")

    scored_test = score_pairs(
        pairs=test_pairs,
        text_model=text_model,
        image_model=image_model,
        max_images_per_pair=args.max_images_per_pair,
    )

    test_metrics = compute_metrics(scored_test, best_threshold)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    save_scored_jsonl(args.output_dir / "scored_val.jsonl", scored_val)
    save_scored_jsonl(args.output_dir / "scored_test.jsonl", scored_test)

    write_json(args.output_dir / "val_metrics.json", val_metrics)
    write_json(args.output_dir / "test_metrics.json", test_metrics)

    print("[INFO] Validation metrics:")
    print(json.dumps(val_metrics, indent=2, ensure_ascii=False))

    print("[INFO] Test metrics:")
    print(json.dumps(test_metrics, indent=2, ensure_ascii=False))

    print(f"[INFO] Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()