from __future__ import annotations

import argparse
import json
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from transformers import CLIPModel, CLIPProcessor

from src.utils.io import read_table, write_json
from src.utils.metrics import summarize_retrieval


def load_splits(path: str) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@torch.no_grad()
def embed_texts(
    model: CLIPModel,
    processor: CLIPProcessor,
    texts: List[str],
    device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding texts"):
        batch = texts[i : i + batch_size]
        inputs = processor(text=batch, images=None, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items() if k != "pixel_values"}
        feats = model.get_text_features(**inputs)
        feats = feats / (feats.norm(dim=1, keepdim=True) + 1e-9)
        embs.append(feats.cpu().numpy())
    return np.vstack(embs)


@torch.no_grad()
def embed_listings_mean_pool(
    model: CLIPModel,
    processor: CLIPProcessor,
    listings: List[Tuple[str, List[str]]],
    device,
    batch_size: int,
) -> Tuple[List[str], np.ndarray]:
    """
    One embedding per listing = mean(embeddings of all listing images)
    """
    model.eval()

    flat: List[Tuple[int, str]] = []
    listing_ids: List[str] = []
    for idx, (lid, paths) in enumerate(listings):
        listing_ids.append(lid)
        for p in paths:
            flat.append((idx, p))

    if len(flat) == 0:
        raise RuntimeError("No images found to embed.")

    sums = None
    counts = np.zeros(len(listings), dtype=np.int64)

    for i in tqdm(range(0, len(flat), batch_size), desc="Embedding listing images (mean-pool)"):
        batch = flat[i : i + batch_size]

        images = []
        idxs = []

        for li, p in batch:
            try:
                images.append(Image.open(p).convert("RGB"))
                idxs.append(li)
            except Exception:
                # skip unreadable images
                continue

        if len(images) == 0:
            continue

        inputs = processor(text=None, images=images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        feats = model.get_image_features(**inputs)  # [b, d]
        feats = feats / (feats.norm(dim=1, keepdim=True) + 1e-9)
        feats = feats.cpu().numpy()

        if sums is None:
            sums = np.zeros((len(listings), feats.shape[1]), dtype=np.float32)

        for row_idx, li in enumerate(idxs):
            sums[li] += feats[row_idx]
            counts[li] += 1

    if sums is None:
        raise RuntimeError("Failed to embed any images. Check image paths/formats.")

    pooled = sums / (counts[:, None] + 1e-9)
    pooled = pooled / (np.linalg.norm(pooled, axis=1, keepdims=True) + 1e-9)

    return listing_ids, pooled


def build_listing_text_table(df) -> Tuple[List[str], List[str]]:
    listing_text = df.groupby("listing_id")["text"].first().reset_index()
    return listing_text["listing_id"].tolist(), listing_text["text"].fillna("").tolist()


def compute_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y_true, y_score))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=str)
    parser.add_argument("--splits", required=True, type=str)
    parser.add_argument("--out", required=True, type=str)
    parser.add_argument("--clip_model", default="laion/CLIP-ViT-B-32-multilingual-v1", type=str)
    parser.add_argument("--k", nargs="+", type=int, default=[1, 5, 10])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--do_mismatch_auc", action="store_true")
    args = parser.parse_args()

    df = read_table(args.manifest)

    if "listing_id" not in df.columns:
        raise ValueError("Manifest must contain 'listing_id'.")
    if "image_paths" not in df.columns:
        raise ValueError("Manifest must contain 'image_paths' (list of images per listing).")

    df["listing_id"] = df["listing_id"].astype(str)
    df["image_paths"] = df["image_paths"].apply(lambda x: list(x) if isinstance(x, (list, tuple)) else x)

    splits = load_splits(args.splits)
    train_ids = set(map(str, splits["train"]))
    test_ids = set(map(str, splits["test"]))

    train_df = df[df["listing_id"].isin(train_ids)].copy()
    test_df = df[df["listing_id"].isin(test_ids)].copy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CLIPModel.from_pretrained(args.clip_model).to(device)
    processor = CLIPProcessor.from_pretrained(args.clip_model)

    # --- Text tables (listing-level) ---
    train_listing_ids_text, train_texts = build_listing_text_table(train_df)
    test_listing_ids_text, test_texts = build_listing_text_table(test_df)

    train_text_emb = embed_texts(model, processor, train_texts, device, args.batch_size)
    test_text_emb = embed_texts(model, processor, test_texts, device, args.batch_size)

    # --- Image embeddings (listing-level mean-pool over ALL images) ---
    train_listings = list(zip(train_df["listing_id"].tolist(), train_df["image_paths"].tolist()))
    test_listings = list(zip(test_df["listing_id"].tolist(), test_df["image_paths"].tolist()))

    train_listing_ids_img, train_img_emb = embed_listings_mean_pool(
        model, processor, train_listings, device, args.batch_size
    )
    test_listing_ids_img, test_img_emb = embed_listings_mean_pool(
        model, processor, test_listings, device, args.batch_size
    )

    # Sanity: the listing IDs for text and image should match the df ordering, but not guaranteed
    # We'll build mapping indices to align results by true_id sets.

    # =======================
    # Text -> Image retrieval
    # =======================
    # Query: test texts (listing-level)
    # Gallery: train listing images (listing-level)

    sim_t2i = test_text_emb @ train_img_emb.T  # [n_test_listings, n_train_listings]
    results_t2i = []
    for i, true_lid in enumerate(test_listing_ids_text):
        ranked_idx = np.argsort(-sim_t2i[i])
        ranked_listing_ids = [train_listing_ids_img[j] for j in ranked_idx]
        results_t2i.append((true_lid, ranked_listing_ids))
    metrics_t2i = summarize_retrieval(results_t2i, ks=args.k)

    # =======================
    # Image -> Text retrieval
    # =======================
    # Query: test listing images (listing-level)
    # Gallery: train texts (listing-level)

    sim_i2t = test_img_emb @ train_text_emb.T  # [n_test_listings, n_train_listings]
    results_i2t = []
    for i, true_lid in enumerate(test_listing_ids_img):
        ranked_idx = np.argsort(-sim_i2t[i])
        ranked_listing_ids = [train_listing_ids_text[j] for j in ranked_idx]
        results_i2t.append((true_lid, ranked_listing_ids))
    metrics_i2t = summarize_retrieval(results_i2t, ks=args.k)

    out = {
        "baseline": "multilingual_clip_listing_meanpool",
        "clip_model": args.clip_model,
        "device": str(device),
        "k": args.k,
        "text_to_image_listing_metrics": metrics_t2i,
        "image_to_text_listing_metrics": metrics_i2t,
        "n_train_listings": len(train_listing_ids_text),
        "n_test_listings": len(test_listing_ids_text),
        "pooling": "mean_over_all_images",
    }

    # =======================
    # Optional mismatch AUC
    # =======================
    
    if args.do_mismatch_auc:
        # For each test listing: positive score = dot(text_emb, its own listing_img_emb)
        # Negative = dot(text_emb, random other listing_img_emb)
        rng = np.random.default_rng(42)

        # Build mapping listing_id -> index for test image embeddings
        idx_by_lid_img = {lid: i for i, lid in enumerate(test_listing_ids_img)}
        idx_by_lid_txt = {lid: i for i, lid in enumerate(test_listing_ids_text)}

        common_lids = sorted(set(test_listing_ids_img).intersection(set(test_listing_ids_text)))
        if len(common_lids) < 2:
            raise RuntimeError("Need at least 2 overlapping test listings for mismatch AUC.")

        y_true = []
        y_score = []

        for lid in common_lids:
            t = test_text_emb[idx_by_lid_txt[lid]]          # (d,)
            pos_img = test_img_emb[idx_by_lid_img[lid]]     # (d,)
            y_true.append(1)
            y_score.append(float(pos_img @ t))

            neg_lid = lid
            while neg_lid == lid:
                neg_lid = common_lids[int(rng.integers(0, len(common_lids)))]
            neg_img = test_img_emb[idx_by_lid_img[neg_lid]]
            y_true.append(0)
            y_score.append(float(neg_img @ t))

        out["mismatch_auc_synthetic"] = compute_auc(np.array(y_true), np.array(y_score))

    write_json(args.out, out)
    print(f"Wrote report: {args.out}")
    print("Text→Image:", metrics_t2i)
    print("Image→Text:", metrics_i2t)
    if "mismatch_auc_synthetic" in out:
        print("Mismatch AUC (synthetic):", out["mismatch_auc_synthetic"])


if __name__ == "__main__":
    main()