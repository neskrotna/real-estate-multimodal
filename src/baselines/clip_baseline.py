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


def unique_rank(ids: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in ids:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


@torch.no_grad()
def embed_texts(model: CLIPModel, processor: CLIPProcessor, texts: List[str], device, batch_size: int) -> np.ndarray:
    model.eval()
    embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding texts"):
        batch = texts[i:i+batch_size]
        inputs = processor(text=batch, images=None, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items() if k != "pixel_values"}
        feats = model.get_text_features(**inputs)
        feats = feats / (feats.norm(dim=1, keepdim=True) + 1e-9)
        embs.append(feats.cpu().numpy())
    return np.vstack(embs)


@torch.no_grad()
def embed_images(model: CLIPModel, processor: CLIPProcessor, image_paths: List[str], device, batch_size: int) -> np.ndarray:
    model.eval()
    embs = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Embedding images"):
        batch_paths = image_paths[i:i+batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = processor(text=None, images=images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        feats = model.get_image_features(**inputs)
        feats = feats / (feats.norm(dim=1, keepdim=True) + 1e-9)
        embs.append(feats.cpu().numpy())
    return np.vstack(embs)


def build_listing_text_table(df) -> Tuple[List[str], List[str]]:
    listing_text = df.groupby("listing_id")["text"].first().reset_index()
    return listing_text["listing_id"].tolist(), listing_text["text"].fillna("").tolist()


def build_query_images(df) -> Tuple[List[str], List[str]]:
    # first image per listing
    q = (
        df.sort_values(["listing_id", "image_index"])
        .groupby("listing_id")
        .first()
        .reset_index()
    )
    return q["listing_id"].tolist(), q["image_path"].tolist()


def compute_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    # simple AUC without sklearn dependency (but sklearn is installed anyway)
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
    df["listing_id"] = df["listing_id"].astype(str)

    splits = load_splits(args.splits)
    train_ids = set(map(str, splits["train"]))
    test_ids = set(map(str, splits["test"]))

    train_df = df[df["listing_id"].isin(train_ids)].copy()
    test_df = df[df["listing_id"].isin(test_ids)].copy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CLIPModel.from_pretrained(args.clip_model).to(device)
    processor = CLIPProcessor.from_pretrained(args.clip_model)

    # --- Build train galleries ---
    train_listing_ids_text, train_texts = build_listing_text_table(train_df)
    train_images = train_df["image_path"].tolist()
    train_image_listing_ids = train_df["listing_id"].tolist()

    train_text_emb = embed_texts(model, processor, train_texts, device, args.batch_size)
    train_img_emb = embed_images(model, processor, train_images, device, args.batch_size)

    # --- Build test queries ---
    test_listing_ids_text, test_texts = build_listing_text_table(test_df)
    test_listing_ids_img, test_img_paths = build_query_images(test_df)

    test_text_emb = embed_texts(model, processor, test_texts, device, args.batch_size)
    test_img_emb = embed_images(model, processor, test_img_paths, device, args.batch_size)

    # =======================
    # Text -> Image retrieval
    # =======================
    sim_t2i = test_text_emb @ train_img_emb.T  # [n_test_listings, n_train_images]
    results_t2i = []
    for i, true_lid in enumerate(test_listing_ids_text):
        ranked_idx = np.argsort(-sim_t2i[i])
        ranked_img_listing_ids = [train_image_listing_ids[j] for j in ranked_idx]
        ranked_listing_ids = unique_rank(ranked_img_listing_ids)
        results_t2i.append((true_lid, ranked_listing_ids))

    metrics_t2i = summarize_retrieval(results_t2i, ks=args.k)

    # =======================
    # Image -> Text retrieval
    # =======================
    sim_i2t = test_img_emb @ train_text_emb.T  # [n_test_listings, n_train_listings]
    results_i2t = []
    for i, true_lid in enumerate(test_listing_ids_img):
        ranked_idx = np.argsort(-sim_i2t[i])
        ranked_listing_ids = [train_listing_ids_text[j] for j in ranked_idx]
        results_i2t.append((true_lid, ranked_listing_ids))

    metrics_i2t = summarize_retrieval(results_i2t, ks=args.k)

    out = {
        "baseline": "multilingual_clip",
        "clip_model": args.clip_model,
        "device": str(device),
        "k": args.k,
        "text_to_image_listing_metrics": metrics_t2i,
        "image_to_text_listing_metrics": metrics_i2t,
        "n_train_listings": len(train_listing_ids_text),
        "n_train_images": len(train_images),
        "n_test_listings": len(test_listing_ids_text),
    }

    # =======================
    # Optional mismatch AUC
    # =======================
    if args.do_mismatch_auc:
        # For each test listing: score = max similarity(text, its images)
        # Then create synthetic mismatches by pairing text with images from another random listing.
        rng = np.random.default_rng(42)

        # Build per-listing image embeddings for test (use ALL images for each listing)
        # We'll embed all test images once.
        all_test_images = test_df["image_path"].tolist()
        all_test_image_listing_ids = test_df["listing_id"].tolist()
        all_test_img_emb = embed_images(model, processor, all_test_images, device, args.batch_size)

        # Map listing -> indices of its images
        idx_by_lid: Dict[str, List[int]] = {}
        for idx, lid in enumerate(all_test_image_listing_ids):
            idx_by_lid.setdefault(lid, []).append(idx)

        # Build text embeddings aligned to listing ids
        lid_to_text_emb = {lid: emb for lid, emb in zip(test_listing_ids_text, test_text_emb)}

        y_true = []
        y_score = []

        lids = test_listing_ids_text
        for lid in lids:
            t = lid_to_text_emb[lid]  # (d,)
            # positive score: best match among its images
            pos_idxs = idx_by_lid[lid]
            pos_scores = all_test_img_emb[pos_idxs] @ t
            y_true.append(1)
            y_score.append(float(np.max(pos_scores)))

            # negative: pick another listing's images
            neg_lid = lid
            while neg_lid == lid:
                neg_lid = lids[int(rng.integers(0, len(lids)))]
            neg_idxs = idx_by_lid[neg_lid]
            neg_scores = all_test_img_emb[neg_idxs] @ t
            y_true.append(0)
            y_score.append(float(np.max(neg_scores)))

        auc = compute_auc(np.array(y_true), np.array(y_score))
        out["mismatch_auc_synthetic"] = auc

    write_json(args.out, out)
    print(f"Wrote report: {args.out}")
    print("Text→Image:", metrics_t2i)
    print("Image→Text:", metrics_i2t)
    if "mismatch_auc_synthetic" in out:
        print("Mismatch AUC (synthetic):", out["mismatch_auc_synthetic"])


if __name__ == "__main__":
    main()