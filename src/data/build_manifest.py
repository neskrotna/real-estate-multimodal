from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from tqdm import tqdm

from src.utils.io import read_jsonl, read_table, write_parquet
from src.utils.image import list_images
from src.utils.text import normalize_text


def load_metadata(input_dir: Path) -> pd.DataFrame:
    """
    Supports:
    - listings.jsonl
    - listings.csv

    Required:
    - listing_id (or id)
    - title
    - description
    """
    jsonl_path = input_dir / "listings.jsonl"
    csv_path = input_dir / "listings.csv"

    if jsonl_path.exists():
        items = read_jsonl(jsonl_path)
        df = pd.DataFrame(items)
    elif csv_path.exists():
        df = read_table(csv_path)
    else:
        raise FileNotFoundError(
            f"Expected {jsonl_path} or {csv_path}. Adapt load_metadata() if different."
        )

    if "listing_id" not in df.columns:
        if "id" in df.columns:
            df = df.rename(columns={"id": "listing_id"})
        else:
            raise ValueError("Metadata must contain 'listing_id' or 'id'.")

    for col in ["title", "description"]:
        if col not in df.columns:
            df[col] = ""

    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--images_dir_name", default="images", type=str)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    images_root = input_dir / args.images_dir_name

    meta = load_metadata(input_dir)

    rows: List[Dict[str, Any]] = []
    for _, row in tqdm(meta.iterrows(), total=len(meta), desc="Building manifest"):
        listing_id = str(row["listing_id"])
        title = row.get("title", "") or ""
        description = row.get("description", "") or ""
        text = normalize_text(title, description)

        listing_img_dir = images_root / listing_id
        imgs = list_images(listing_img_dir)
        if len(imgs) == 0:
            continue

        for idx, img_path in enumerate(imgs):
            rows.append({
                "listing_id": listing_id,
                "image_path": img_path.as_posix(),
                "image_index": idx,
                "title": title,
                "description": description,
                "text": text,

                "city": row.get("city", None),
                "district": row.get("district", None),
                "rooms": row.get("rooms", None),
                "sqm": row.get("sqm", None),
                "price": row.get("price", None),
                "source": row.get("source", None),
            })

    manifest = pd.DataFrame(rows)
    if len(manifest) == 0:
        raise RuntimeError("Manifest is empty. Check images folder and listing_id mapping.")

    write_parquet(manifest, args.output)
    print(f"✅ Wrote manifest: {args.output}")
    print(f"Listings: {manifest['listing_id'].nunique()} | Images: {len(manifest)}")


if __name__ == "__main__":
    main()