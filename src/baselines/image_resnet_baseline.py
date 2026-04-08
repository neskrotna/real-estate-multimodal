from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms

from src.utils.io import read_json, read_jsonl, write_json, write_jsonl
from src.utils.metrics import summarize_retrieval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ResNet image-only baseline: image-half -> image-half retrieval"
    )
    parser.add_argument("--listings", type=str, default="data/processed/listings.jsonl")
    parser.add_argument("--split", type=str, default="data/processed/split_v1.json")
    parser.add_argument("--image-root", type=str, default="data/processed/images")
    parser.add_argument("--run-dir", type=str, default="runs/image_resnet_half_to_half")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_test_records(listings_path: str, split_path: str) -> List[dict]:
    rows = read_jsonl(listings_path)
    split_map = read_json(split_path)
    test_ids = set(split_map.get("test", []))

    usable = []
    for row in rows:
        listing_id = str(row.get("listing_id", "")).strip()
        if listing_id not in test_ids:
            continue

        image_folder = str(row.get("image_folder", "")).strip()
        image_count = int(row.get("image_count", 0))

        if not image_folder or image_count < 2:
            continue

        usable.append(
            {
                "listing_id": listing_id,
                "image_folder": image_folder,
                "image_count": image_count,
            }
        )

    return usable


def build_listing_image_paths(image_root: str, image_folder: str, image_count: int) -> List[str]:
    folder = Path(image_root) / image_folder
    paths = []

    for i in range(1, image_count + 1):
        p = folder / f"{image_folder}_img_{i:02d}.jpg"
        if p.exists():
            paths.append(str(p))

    return paths


def split_image_paths(paths: List[str]) -> tuple[List[str], List[str]]:
    if len(paths) < 2:
        return [], []

    mid = len(paths) // 2
    query_paths = paths[:mid]
    candidate_paths = paths[mid:]

    if len(query_paths) == 0:
        query_paths = [paths[0]]
    if len(candidate_paths) == 0:
        candidate_paths = [paths[-1]]

    return query_paths, candidate_paths


def build_model(device: str):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model.eval()
    model.to(device)
    return model


def build_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


@torch.no_grad()
def encode_image_batch(model, image_paths: List[str], transform, device: str) -> np.ndarray:
    tensors = [transform(load_image(p)) for p in image_paths]
    batch = torch.stack(tensors).to(device)
    emb = model(batch).cpu().numpy()
    return emb


@torch.no_grad()
def encode_listing_images(
    model,
    image_paths: List[str],
    transform,
    device: str,
    batch_size: int,
) -> np.ndarray:
    if not image_paths:
        raise ValueError("No image paths provided")

    chunks = []
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start:start + batch_size]
        batch_emb = encode_image_batch(model, batch_paths, transform, device)
        chunks.append(batch_emb)

    all_emb = np.concatenate(chunks, axis=0)
    return all_emb.mean(axis=0)


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
    b_norm = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-12, None)
    return a_norm @ b_norm.T


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    records = load_test_records(args.listings, args.split)

    if len(records) == 0:
        raise ValueError("No usable test records found with at least 2 images.")

    model = build_model(args.device)
    transform = build_transform()

    query_ids = []
    candidate_ids = []
    query_embs = []
    candidate_embs = []

    for record in records:
        all_paths = build_listing_image_paths(
            image_root=args.image_root,
            image_folder=record["image_folder"],
            image_count=record["image_count"],
        )

        if len(all_paths) < 2:
            continue

        query_paths, candidate_paths = split_image_paths(all_paths)

        if not query_paths or not candidate_paths:
            continue

        query_emb = encode_listing_images(
            model=model,
            image_paths=query_paths,
            transform=transform,
            device=args.device,
            batch_size=args.batch_size,
        )

        candidate_emb = encode_listing_images(
            model=model,
            image_paths=candidate_paths,
            transform=transform,
            device=args.device,
            batch_size=args.batch_size,
        )

        query_ids.append(record["listing_id"])
        candidate_ids.append(record["listing_id"])
        query_embs.append(query_emb)
        candidate_embs.append(candidate_emb)

    if len(query_embs) == 0:
        raise ValueError("No usable embeddings could be created.")

    query_embs = np.vstack(query_embs)
    candidate_embs = np.vstack(candidate_embs)

    similarity = cosine_similarity_matrix(query_embs, candidate_embs)

    results = []
    top5_rows = []

    for i, true_id in enumerate(query_ids):
        row_scores = similarity[i]
        ranked_idx = np.argsort(-row_scores)
        ranked_ids = [candidate_ids[j] for j in ranked_idx]

        results.append((true_id, ranked_ids))

        top5_rows.append(
            {
                "query_id": true_id,
                "predictions": [
                    {
                        "rank": rank + 1,
                        "candidate_id": candidate_ids[j],
                        "score": float(row_scores[j]),
                    }
                    for rank, j in enumerate(ranked_idx[:5])
                ],
            }
        )

    metrics = summarize_retrieval(results, ks=[1, 5, 10])

    write_json(run_dir / "metrics.json", metrics)
    write_jsonl(run_dir / "top5_predictions.jsonl", top5_rows)

    print("Image ResNet half->half baseline finished.")
    print(metrics)


if __name__ == "__main__":
    main()