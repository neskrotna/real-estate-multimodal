from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from yaml import loader

from src.utils.io import read_json, read_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 8: pair classifier on CLIP embeddings."
    )
    parser.add_argument("--train-pairs", type=Path, default=Path("data/processed/pairs_train.jsonl"))
    parser.add_argument("--val-pairs", type=Path, default=Path("data/processed/pairs_val.jsonl"))
    parser.add_argument("--test-pairs", type=Path, default=Path("data/processed/pairs_test.jsonl"))
    parser.add_argument("--text-model", type=str, default="sentence-transformers/clip-ViT-B-32-multilingual-v1")
    parser.add_argument("--image-model", type=str, default="clip-ViT-B-32")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/exp_8_clip_pair_classifier"))
    parser.add_argument("--max-images-per-pair", type=int, default=4)
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=6)
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
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


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
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

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


class PairDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class PairClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Loading multilingual text model: {args.text_model}")
    print(f"[INFO] Loading image model: {args.image_model}")

    text_model = SentenceTransformer(args.text_model, device=str(device))
    image_model = SentenceTransformer(args.image_model, device=str(device))

    train_pairs = read_jsonl(args.train_pairs)
    val_pairs = read_jsonl(args.val_pairs)
    test_pairs = read_jsonl(args.test_pairs)

    text_cache: dict[str, np.ndarray] = {}
    image_cache: dict[str, dict[str, Any]] = {}

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

    def get_image_embeddings(pair: dict[str, Any]) -> dict[str, Any]:
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
            image_cache[lid] = {
                "paths": selected_paths,
                "embs": embs,
            }
        return image_cache[lid]

    def build_features(pairs: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
        X = []
        y = []
        rows_out = []

        for pair in tqdm(pairs, desc="Building pair-classifier features"):
            text_emb = get_text_embedding(pair)
            image_data = get_image_embeddings(pair)
            image_embs = image_data["embs"]

            pooled_image = np.mean(image_embs, axis=0)
            pooled_image = pooled_image / np.linalg.norm(pooled_image)

            sims = image_embs @ text_emb
            max_sim = float(np.max(sims))
            mean_sim = float(np.mean(sims))
            top2_mean = float(np.mean(np.sort(sims)[::-1][:2])) if len(sims) >= 2 else max_sim
            min_sim = float(np.min(sims))
            std_sim = float(np.std(sims))

            abs_diff = np.abs(text_emb - pooled_image)
            prod = text_emb * pooled_image

            features = np.concatenate(
                [
                    text_emb,
                    pooled_image,
                    abs_diff,
                    prod,
                    np.array([max_sim, mean_sim, top2_mean, min_sim, std_sim], dtype=np.float32),
                ]
            )

            X.append(features)
            y.append(pair["label"])

            row = dict(pair)
            row["clip_max_similarity"] = max_sim
            row["clip_mean_similarity"] = mean_sim
            row["clip_top2_mean_similarity"] = top2_mean
            rows_out.append(row)

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), rows_out

    X_train, y_train, _ = build_features(train_pairs)
    X_val, y_val, val_rows = build_features(val_pairs)
    X_test, y_test, test_rows = build_features(test_pairs)

    train_dataset = PairDataset(X_train, y_train)
    val_dataset = PairDataset(X_val, y_val)
    test_dataset = PairDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    input_dim = X_train.shape[1]
    model = PairClassifier(input_dim=input_dim, hidden_dim=args.hidden_dim).to(device)

    pos_weight_value = float((len(y_train) - y_train.sum()) / y_train.sum()) if y_train.sum() > 0 else 1.0
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_value, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                val_losses.append(loss.item())

        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        val_loss = float(np.mean(val_losses)) if val_losses else 0.0
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        print(f"[INFO] Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print("[INFO] Early stopping triggered.")
            break

    if best_state is None:
        raise RuntimeError("No best model state saved.")

    model.load_state_dict(best_state)
    model.to(device)
    model.eval()

    def predict_probs(loader: DataLoader) -> np.ndarray:
        probs = []
        with torch.no_grad():
            for batch_x, _ in loader:
                batch_x = batch_x.to(device)
                logits = model(batch_x)
                batch_prob = torch.sigmoid(logits).cpu().numpy()
                probs.extend(batch_prob.tolist())
        return np.array(probs, dtype=np.float32)

    val_prob = predict_probs(val_loader)
    test_prob = predict_probs(test_loader)

    best_threshold, best_val_f1 = find_best_threshold(y_val, val_prob)

    val_metrics = compute_metrics(y_val, val_prob, best_threshold)
    test_metrics = compute_metrics(y_test, test_prob, best_threshold)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, args.output_dir / "best_model.pt")
    write_json(args.output_dir / "history.json", history)
    write_json(args.output_dir / "val_metrics.json", val_metrics)
    write_json(args.output_dir / "test_metrics.json", test_metrics)

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