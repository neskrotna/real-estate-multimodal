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

from src.utils.io import read_json, read_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 7: hard-negative fine-tuning with triplet-style loss."
    )
    parser.add_argument("--listings", type=Path, default=Path("data/processed/listings.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/manifest.jsonl"))
    parser.add_argument("--train-pairs", type=Path, default=Path("data/processed/pairs_train.jsonl"))
    parser.add_argument("--val-pairs", type=Path, default=Path("data/processed/pairs_val.jsonl"))
    parser.add_argument("--test-pairs", type=Path, default=Path("data/processed/pairs_test.jsonl"))
    parser.add_argument("--text-model", type=str, default="sentence-transformers/clip-ViT-B-32-multilingual-v1")
    parser.add_argument("--image-model", type=str, default="clip-ViT-B-32")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/exp_7_clip_hard_negative_finetune"))
    parser.add_argument("--max-images-per-listing", type=int, default=4)
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=5)
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
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


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
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

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


class TripletDataset(Dataset):
    def __init__(
        self,
        triplets: list[tuple[str, str, str]],
        text_embs: dict[str, np.ndarray],
        pooled_image_embs: dict[str, np.ndarray],
    ):
        self.triplets = triplets
        self.text_embs = text_embs
        self.pooled_image_embs = pooled_image_embs

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        text_id, pos_id, neg_id = self.triplets[idx]
        return (
            torch.tensor(self.text_embs[text_id], dtype=torch.float32),
            torch.tensor(self.pooled_image_embs[pos_id], dtype=torch.float32),
            torch.tensor(self.pooled_image_embs[neg_id], dtype=torch.float32),
        )


class ProjectionModel(nn.Module):
    def __init__(self, embedding_dim: int = 512, projection_dim: int = 256):
        super().__init__()
        self.text_head = nn.Sequential(
            nn.Linear(embedding_dim, projection_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(projection_dim, projection_dim),
        )
        self.image_head = nn.Sequential(
            nn.Linear(embedding_dim, projection_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(projection_dim, projection_dim),
        )

    def project_text(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.text_head(x), p=2, dim=-1)

    def project_image(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.image_head(x), p=2, dim=-1)


def triplet_margin_loss(anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, margin: float) -> torch.Tensor:
    sim_pos = torch.sum(anchor * positive, dim=-1)
    sim_neg = torch.sum(anchor * negative, dim=-1)
    return torch.relu(margin - sim_pos + sim_neg).mean()


def build_triplets(pairs: list[dict[str, Any]]) -> list[tuple[str, str, str]]:
    triplets = []
    for row in pairs:
        if row.get("pair_type") == "hard_negative":
            text_id = row["text_listing_id"]
            pos_id = row["text_listing_id"]
            neg_id = row["image_listing_id"]
            triplets.append((text_id, pos_id, neg_id))
    return triplets


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Loading multilingual text model: {args.text_model}")
    print(f"[INFO] Loading image model: {args.image_model}")

    text_model = SentenceTransformer(args.text_model, device=str(device))
    image_model = SentenceTransformer(args.image_model, device=str(device))

    listings = read_jsonl(args.listings)
    manifest = read_jsonl(args.manifest)
    train_pairs = read_jsonl(args.train_pairs)
    val_pairs = read_jsonl(args.val_pairs)
    test_pairs = read_jsonl(args.test_pairs)

    listings_by_id = {row["listing_id"]: row for row in listings}
    manifest_by_id = {row["listing_id"]: row for row in manifest}
    all_ids = sorted(set(listings_by_id.keys()))

    text_embs: dict[str, np.ndarray] = {}
    pooled_image_embs: dict[str, np.ndarray] = {}

    print("[INFO] Precomputing frozen text embeddings...")
    for lid in tqdm(all_ids):
        text = listings_by_id[lid]["description"]
        emb = text_model.encode([text], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)[0]
        text_embs[lid] = emb

    print("[INFO] Precomputing frozen pooled image embeddings...")
    for lid in tqdm(all_ids):
        image_paths = manifest_by_id[lid]["image_paths"][: args.max_images_per_listing]
        images = [load_image(p) for p in image_paths]
        embs = image_model.encode(images, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        pooled = np.mean(embs, axis=0)
        pooled = pooled / np.linalg.norm(pooled)
        pooled_image_embs[lid] = pooled

    train_triplets = build_triplets(train_pairs)
    val_triplets = build_triplets(val_pairs)

    if len(train_triplets) == 0 or len(val_triplets) == 0:
        raise ValueError("No hard-negative triplets found. Check pair generation.")

    train_dataset = TripletDataset(train_triplets, text_embs, pooled_image_embs)
    val_dataset = TripletDataset(val_triplets, text_embs, pooled_image_embs)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = ProjectionModel(args.embedding_dim, args.projection_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []

        for text_x, pos_x, neg_x in train_loader:
            text_x = text_x.to(device)
            pos_x = pos_x.to(device)
            neg_x = neg_x.to(device)

            optimizer.zero_grad()
            anchor = model.project_text(text_x)
            positive = model.project_image(pos_x)
            negative = model.project_image(neg_x)

            loss = triplet_margin_loss(anchor, positive, negative, args.margin)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []

        with torch.no_grad():
            for text_x, pos_x, neg_x in val_loader:
                text_x = text_x.to(device)
                pos_x = pos_x.to(device)
                neg_x = neg_x.to(device)

                anchor = model.project_text(text_x)
                positive = model.project_image(pos_x)
                negative = model.project_image(neg_x)

                loss = triplet_margin_loss(anchor, positive, negative, args.margin)
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

    projected_text_cache: dict[str, np.ndarray] = {}
    projected_image_cache: dict[str, np.ndarray] = {}

    def project_text_listing(lid: str) -> np.ndarray:
        if lid not in projected_text_cache:
            with torch.no_grad():
                x = torch.tensor(text_embs[lid], dtype=torch.float32, device=device).unsqueeze(0)
                z = model.project_text(x)
                projected_text_cache[lid] = z.squeeze(0).cpu().numpy()
        return projected_text_cache[lid]

    def project_image_listing(lid: str) -> np.ndarray:
        if lid not in projected_image_cache:
            with torch.no_grad():
                x = torch.tensor(pooled_image_embs[lid], dtype=torch.float32, device=device).unsqueeze(0)
                z = model.project_image(x)
                projected_image_cache[lid] = z.squeeze(0).cpu().numpy()
        return projected_image_cache[lid]

    def score_pairs(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        scored = []
        for pair in tqdm(pairs, desc="Scoring pairs"):
            text_vec = project_text_listing(pair["text_listing_id"])
            image_vec = project_image_listing(pair["image_listing_id"])

            sim = float(np.dot(text_vec, image_vec))

            row = dict(pair)
            row["similarity"] = sim
            scored.append(row)
        return scored

    scored_val = score_pairs(val_pairs)
    best_threshold, best_val_f1 = find_best_threshold(scored_val)
    val_metrics = compute_metrics(scored_val, best_threshold)

    scored_test = score_pairs(test_pairs)
    test_metrics = compute_metrics(scored_test, best_threshold)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, args.output_dir / "best_model.pt")
    write_json(args.output_dir / "history.json", history)
    write_json(args.output_dir / "val_metrics.json", val_metrics)
    write_json(args.output_dir / "test_metrics.json", test_metrics)
    write_jsonl(args.output_dir / "scored_val.jsonl", scored_val)
    write_jsonl(args.output_dir / "scored_test.jsonl", scored_test)

    print(f"[INFO] Best validation threshold: {best_threshold:.2f}")
    print(f"[INFO] Best validation F1: {best_val_f1:.4f}")
    print("[INFO] Validation metrics:")
    print(json.dumps(val_metrics, indent=2, ensure_ascii=False))
    print("[INFO] Test metrics:")
    print(json.dumps(test_metrics, indent=2, ensure_ascii=False))
    print(f"[INFO] Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()