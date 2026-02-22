from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from tqdm import tqdm

from src.utils.io import read_table, write_parquet
from src.utils.image import list_images
from src.utils.text import normalize_text


def load_metadata(input_dir: Path) -> pd.DataFrame:
    """
    Expects:
        data/raw/
            ├── descriptions/listings.xlsx
            └── images/
    """

    descriptions_dir = input_dir / "descriptions"

    if not descriptions_dir.exists():
        raise FileNotFoundError(f"{descriptions_dir} not found.")

    # Look for Excel or CSV inside descriptions/
    candidates = list(descriptions_dir.glob("*.xlsx")) + list(descriptions_dir.glob("*.csv"))
    if len(candidates) == 0:
        raise FileNotFoundError("No metadata file (.xlsx or .csv) found in descriptions/")

    meta_path = candidates[0]  # assume single file
    print(f"Using metadata file: {meta_path}")

    df = read_table(meta_path)

    # Normalize ID column
    if "listing_id" not in df.columns:
        if "anzeigen_id" in df.columns:
            df = df.rename(columns={"anzeigen_id": "listing_id"})
        else:
            raise ValueError("Metadata must contain 'anzeigen_id' or 'listing_id'.")

    # Normalize title/description
    if "title" not in df.columns:
        if "titel" in df.columns:
            df = df.rename(columns={"titel": "title"})
        else:
            df["title"] = ""

    if "description" not in df.columns:
        if "beschreibung" in df.columns:
            df = df.rename(columns={"beschreibung": "description"})
        else:
            df["description"] = ""

    df["listing_id"] = df["listing_id"].astype(str)

    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    images_root = input_dir / "images"

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

        rows.append(
            {
                "listing_id": listing_id,
                "image_paths": [p.as_posix() for p in imgs],
                "n_images": len(imgs),
                "title": title,
                "description": description,
                "text": text,

                # optional metadata
                "city": row.get("stadt", None),
                "rooms": row.get("zimmer", None),
                "sqm": row.get("wohnfläche_m2", None),
                "price": row.get("preis_brutto", None),
                "property_type": row.get("immobilientyp", None),
                "condition": row.get("zustand", None),
            }
        )

    manifest = pd.DataFrame(rows)

    if len(manifest) == 0:
        raise RuntimeError("Manifest is empty. Check images folder and ID matching.")

    write_parquet(manifest, args.output)

    print(f"Wrote manifest: {args.output}")
    print(f"Listings: {manifest['listing_id'].nunique()}")
    print(f"Total images: {manifest['n_images'].sum()}")


if __name__ == "__main__":
    main()