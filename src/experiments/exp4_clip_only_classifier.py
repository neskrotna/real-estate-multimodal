from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.utils.io import read_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 4: CLIP-only classifier using similarity features only."
    )
    parser.add_argument(
        "--train-pairs",
        type=Path,
        default=Path("data/processed/pairs_train.jsonl"),
    )
    parser.add_argument(
        "--val-pairs",
        type=Path,
        default=Path("data/processed/pairs_val.jsonl"),
    )
    parser.add_argument(
        "--test-pairs",
        type=Path,
        default=Path("data/processed/pairs_test.jsonl"),
    )
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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/exp4_clip_only_classifier"),
    )
    parser.add_argument(
        "--max-images-per-pair",
        type=int,
        default=4,
    )
    return parser.parse_args()


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def f1_from_probs(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> float:
    y_pred = (y_prob >= threshold).astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    best_threshold = 0.5
    best_f1 = -1.0

    for i in range(1, 100):
        threshold = i / 100.0
        f1 = f1_from_probs(y_true, y_prob, threshold)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold, best_f1


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    total = len(y_true)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)

    return {
        "num_pairs": int(total),
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def main() -> None:
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Loading multilingual text model: {args.text_model}")
    print(f"[INFO] Loading image model: {args.image_model}")

    text_model = SentenceTransformer(args.text_model, device=device)
    image_model = SentenceTransformer(args.image_model, device=device)

    train_pairs = read_jsonl(args.train_pairs)
    val_pairs = read_jsonl(args.val_pairs)
    test_pairs = read_jsonl(args.test_pairs)

    text_cache: dict[str, np.ndarray] = {}
    image_cache: dict[str, tuple[list[str], np.ndarray]] = {}

    def get_text_embedding(pair: dict[str, Any]) -> np.ndarray:
        lid = pair["text_listing_id"]
        if lid not in text_cache:
            emb = text_model.encode(
                [pair["text"]],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )[0]
            text_cache[lid] = emb
        return text_cache[lid]

    def get_image_embeddings(pair: dict[str, Any]) -> tuple[list[str], np.ndarray]:
        lid = pair["image_listing_id"]
        if lid not in image_cache:
            selected_paths = pair["image_paths"][: args.max_images_per_pair]
            images = [load_image(p) for p in selected_paths]
            embs = image_model.encode(
                images,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            image_cache[lid] = (selected_paths, embs)
        return image_cache[lid]

    feature_names = [
        "clip_max_similarity",
        "clip_mean_similarity",
        "clip_top2_mean_similarity",
    ]

    def build_features(pairs: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
        X = []
        y = []
        rows_out = []

        for pair in tqdm(pairs, desc="Building features"):
            text_emb = get_text_embedding(pair)
            _, image_embs = get_image_embeddings(pair)
            sims = image_embs @ text_emb

            max_sim = float(np.max(sims))
            mean_sim = float(np.mean(sims))
            sorted_sims = np.sort(sims)[::-1]
            top2_mean = float(np.mean(sorted_sims[:2])) if len(sorted_sims) >= 2 else max_sim

            features = [
                max_sim,
                mean_sim,
                top2_mean,
            ]

            X.append(features)
            y.append(pair["label"])

            out_row = dict(pair)
            out_row["clip_max_similarity"] = max_sim
            out_row["clip_mean_similarity"] = mean_sim
            out_row["clip_top2_mean_similarity"] = top2_mean
            rows_out.append(out_row)

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64), rows_out

    X_train, y_train, _ = build_features(train_pairs)
    X_val, y_val, val_rows = build_features(val_pairs)
    X_test, y_test, test_rows = build_features(test_pairs)

    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=3000, random_state=42)),
        ]
    )

    clf.fit(X_train, y_train)

    val_prob = clf.predict_proba(X_val)[:, 1]
    best_threshold, best_val_f1 = find_best_threshold(y_val, val_prob)

    test_prob = clf.predict_proba(X_test)[:, 1]

    val_metrics = compute_metrics(y_val, val_prob, best_threshold)
    test_metrics = compute_metrics(y_test, test_prob, best_threshold)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(clf, args.output_dir / "classifier.joblib")
    write_json(args.output_dir / "feature_names.json", feature_names)
    write_json(args.output_dir / "val_metrics.json", val_metrics)
    write_json(args.output_dir / "test_metrics.json", test_metrics)

    clf_step: LogisticRegression = clf.named_steps["clf"]
    coef_map = {name: float(weight) for name, weight in zip(feature_names, clf_step.coef_[0])}
    write_json(args.output_dir / "feature_coefficients.json", coef_map)

    for row, prob in zip(val_rows, val_prob):
        row["match_probability"] = float(prob)
    for row, prob in zip(test_rows, test_prob):
        row["match_probability"] = float(prob)

    write_jsonl(args.output_dir / "scored_val.jsonl", val_rows)
    write_jsonl(args.output_dir / "scored_test.jsonl", test_rows)

    print(f"[INFO] Best validation threshold: {best_threshold:.2f}")
    print(f"[INFO] Best validation F1: {best_val_f1:.4f}")
    print("[INFO] Validation metrics:")
    print(json.dumps(val_metrics, indent=2, ensure_ascii=False))
    print("[INFO] Test metrics:")
    print(json.dumps(test_metrics, indent=2, ensure_ascii=False))
    print(f"[INFO] Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()