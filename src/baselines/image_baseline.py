from __future__ import annotations

import argparse
import json
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torchvision.transforms as T
import timm

from src.utils.io import read_table, write_json
from src.utils.metrics import summarize_retrieval


def load_splits(path: str) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@torch.no_grad()
def embed_listings_mean_pool(
    listings: List[Tuple[str, List[str]]],
    model,
    transform,
    device,
    batch_size: int,
) -> Tuple[List[str], np.ndarray]:
    """
    Compute one embedding per listing by:
      - embedding all images
      - mean pooling image embeddings for that listing

    Args:
        listings: list of (listing_id, [image_path1, image_path2, ...])
    Returns:
        listing_ids: list[str]
        embeddings: np.ndarray [n_listings, dim]
    """
    model.eval()

    # Flatten all images into a single list with listing index
    flat: List[Tuple[int, str]] = []
    listing_ids: List[str] = []
    for idx, (lid, paths) in enumerate(listings):
        listing_ids.append(lid)
        for p in paths:
            flat.append((idx, p))

    if len(flat) == 0:
        raise RuntimeError("No images found to embed.")

    sums = None  # initialized after first forward pass to know embedding dim
    counts = np.zeros(len(listings), dtype=np.int64)

    for i in tqdm(range(0, len(flat), batch_size), desc="Embedding images (mean-pooling per listing)"):
        batch = flat[i : i + batch_size]

        imgs = []
        idxs = []

        for li, p in batch:
            try:
                img = Image.open(p).convert("RGB")
                imgs.append(transform(img))
                idxs.append(li)
            except Exception:
                # Don't crash the run because of 1 broken file
                continue

        if len(imgs) == 0:
            continue

        x = torch.stack(imgs).to(device)
        y = model(x).detach().cpu().numpy()  # [b, dim]

        if sums is None:
            sums = np.zeros((len(listings), y.shape[1]), dtype=np.float32)

        for row_idx, li in enumerate(idxs):
            sums[li] += y[row_idx]
            counts[li] += 1

    if sums is None:
        raise RuntimeError("Failed to embed any images. Check image paths and formats.")

    pooled = sums / (counts[:, None] + 1e-9)
    pooled = pooled / (np.linalg.norm(pooled, axis=1, keepdims=True) + 1e-9)

    return listing_ids, pooled


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=str)
    parser.add_argument("--splits", required=True, type=str)
    parser.add_argument("--out", required=True, type=str)
    parser.add_argument("--k", nargs="+", type=int, default=[1, 5, 10])
    parser.add_argument("--model", type=str, default="convnext_tiny")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    df = read_table(args.manifest)

    if "listing_id" not in df.columns:
        raise ValueError("Manifest must contain 'listing_id'.")

    if "image_paths" not in df.columns:
        raise ValueError("Manifest must contain 'image_paths' (list of images per listing).")

    df["listing_id"] = df["listing_id"].astype(str)

    # Ensure list type (parquet usually restores list-columns fine, but let's be safe)
    df["image_paths"] = df["image_paths"].apply(
        lambda x: list(x) if isinstance(x, (list, tuple)) else x
    )

    splits = load_splits(args.splits)
    train_ids = set(map(str, splits["train"]))
    test_ids = set(map(str, splits["test"]))

    train_df = df[df["listing_id"].isin(train_ids)].copy()
    test_df = df[df["listing_id"].isin(test_ids)].copy()

    train_listings = list(zip(train_df["listing_id"].tolist(), train_df["image_paths"].tolist()))
    test_listings = list(zip(test_df["listing_id"].tolist(), test_df["image_paths"].tolist()))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backbone = timm.create_model(args.model, pretrained=True, num_classes=0)
    backbone = backbone.to(device)

    transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    train_listing_ids, train_emb = embed_listings_mean_pool(
        train_listings, backbone, transform, device, batch_size=args.batch_size
    )
    test_listing_ids, test_emb = embed_listings_mean_pool(
        test_listings, backbone, transform, device, batch_size=args.batch_size
    )

    sim = test_emb @ train_emb.T  # [n_test, n_train]

    results = []
    for i, true_id in enumerate(test_listing_ids):
        ranked_idx = np.argsort(-sim[i])
        ranked_listing_ids = [train_listing_ids[j] for j in ranked_idx]
        results.append((true_id, ranked_listing_ids))

    summary = summarize_retrieval(results, ks=args.k)

    out = {
        "baseline": f"image_retrieval_listing_meanpool_{args.model}",
        "split": "test_listings_vs_train_listings",
        "k": args.k,
        "metrics": summary,
        "n_train_listings": len(train_listing_ids),
        "n_test_listings": len(test_listing_ids),
        "device": str(device),
        "pooling": "mean_over_all_images",
    }

    write_json(args.out, out)
    print(f"Wrote report: {args.out}")
    print(out["metrics"])


if __name__ == "__main__":
    main()