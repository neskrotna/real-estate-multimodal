from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.io import read_table, write_parquet
from src.utils.seed import set_seed


def load_splits(path: str) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_pairs_for_split(
    manifest: pd.DataFrame,
    listing_ids: List[str],
    neg_per_pos: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    split_df = manifest[manifest["listing_id"].astype(str).isin(set(map(str, listing_ids)))].copy()
    split_df["listing_id"] = split_df["listing_id"].astype(str)

    text_by_listing = split_df.groupby("listing_id")["text"].first().to_dict()
    unique_listings = sorted(text_by_listing.keys())
    if len(unique_listings) < 2:
        raise RuntimeError("Need at least 2 listings in a split to create negatives.")

    rows = []
    for _, r in tqdm(split_df.iterrows(), total=len(split_df), desc="Building pairs"):
        lid = r["listing_id"]
        img = r["image_path"]
        pos_text = text_by_listing[lid]

        rows.append({
            "listing_id": lid,
            "image_path": img,
            "text": pos_text,
            "label": 1,
            "neg_type": None,
            "neg_listing_id": None,
        })

        for _ in range(neg_per_pos):
            neg_lid = lid
            while neg_lid == lid:
                neg_lid = unique_listings[int(rng.integers(0, len(unique_listings)))]
            rows.append({
                "listing_id": lid,          # true listing of the image
                "image_path": img,
                "text": text_by_listing[neg_lid],
                "label": 0,
                "neg_type": "random",
                "neg_listing_id": neg_lid,
            })

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=str)
    parser.add_argument("--splits", required=True, type=str)
    parser.add_argument("--out_dir", required=True, type=str)
    parser.add_argument("--neg_per_pos", default=3, type=int)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    set_seed(args.seed)

    manifest = read_table(args.manifest)
    splits = load_splits(args.splits)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name, seed_offset in [("train", 0), ("val", 1), ("test", 2)]:
        df_pairs = build_pairs_for_split(
            manifest=manifest,
            listing_ids=splits[split_name],
            neg_per_pos=args.neg_per_pos,
            seed=args.seed + seed_offset,
        )
        out_path = out_dir / f"pairs_{split_name}.parquet"
        write_parquet(df_pairs, out_path)
        print(f"Wrote pairs: {out_path} | rows={len(df_pairs)}")


if __name__ == "__main__":
    main()