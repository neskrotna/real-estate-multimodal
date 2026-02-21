from __future__ import annotations

import argparse
import json
from typing import Dict, List

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
def embed_images(paths: List[str], model, transform, device, batch_size: int) -> np.ndarray:
    model.eval()
    embs = []
    for i in tqdm(range(0, len(paths), batch_size), desc="Embedding images"):
        batch_paths = paths[i:i+batch_size]
        imgs = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            imgs.append(transform(img))
        x = torch.stack(imgs).to(device)
        y = model(x).cpu().numpy()
        embs.append(y)
    return np.vstack(embs)


def unique_listing_rank(ranked_listing_ids: List[str]) -> List[str]:
    seen = set()
    out = []
    for lid in ranked_listing_ids:
        if lid in seen:
            continue
        seen.add(lid)
        out.append(lid)
    return out


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
    df["listing_id"] = df["listing_id"].astype(str)

    splits = load_splits(args.splits)
    train_ids = set(map(str, splits["train"]))
    test_ids = set(map(str, splits["test"]))

    train_df = df[df["listing_id"].isin(train_ids)].copy()
    test_df = df[df["listing_id"].isin(test_ids)].copy()

    # Query: first image per test listing
    test_queries = (
        test_df.sort_values(["listing_id", "image_index"])
        .groupby("listing_id")
        .first()
        .reset_index()
    )

    train_images = train_df["image_path"].tolist()
    train_image_listing_ids = train_df["listing_id"].tolist()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backbone = timm.create_model(args.model, pretrained=True, num_classes=0)
    backbone = backbone.to(device)

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    train_emb = embed_images(train_images, backbone, transform, device, batch_size=args.batch_size)
    train_emb = train_emb / (np.linalg.norm(train_emb, axis=1, keepdims=True) + 1e-9)

    test_paths = test_queries["image_path"].tolist()
    test_emb = embed_images(test_paths, backbone, transform, device, batch_size=args.batch_size)
    test_emb = test_emb / (np.linalg.norm(test_emb, axis=1, keepdims=True) + 1e-9)

    sim = test_emb @ train_emb.T

    results = []
    for i, true_id in enumerate(test_queries["listing_id"].tolist()):
        ranked_idx = np.argsort(-sim[i])
        ranked_img_listing_ids = [train_image_listing_ids[j] for j in ranked_idx]
        ranked_listing_ids = unique_listing_rank(ranked_img_listing_ids)
        results.append((true_id, ranked_listing_ids))

    summary = summarize_retrieval(results, ks=args.k)

    out = {
        "baseline": f"image_retrieval_{args.model}",
        "split": "test_query_images_vs_train_images",
        "k": args.k,
        "metrics": summary,
        "n_train_images": len(train_images),
        "n_test_queries": len(test_queries),
        "device": str(device),
    }

    write_json(args.out, out)
    print(f"Wrote report: {args.out}")
    print(out["metrics"])


if __name__ == "__main__":
    main()