from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

LISTINGS_PATH = PROJECT_ROOT / "data" / "processed" / "listings.jsonl"
MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "manifest.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "demo" / "artifacts" / "listings_metadata.json"


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    listings = read_jsonl(LISTINGS_PATH)
    manifest = read_jsonl(MANIFEST_PATH)

    manifest_map = {row["listing_id"]: row for row in manifest}

    merged = []
    for row in listings:
        listing_id = row["listing_id"]
        manifest_row = manifest_map.get(listing_id, {})
        image_paths = manifest_row.get("image_paths", [])
        preview_image = image_paths[0] if image_paths else None

        merged.append(
            {
                "listing_id": listing_id,
                "title": row.get("title", ""),
                "description": row.get("description", ""),
                "city": row.get("city", ""),
                "country": row.get("country", ""),
                "address": row.get("address", ""),
                "postal_code": row.get("postal_code", ""),
                "price_eur": row.get("price_eur"),
                "rooms": row.get("rooms"),
                "area_m2": row.get("area_m2"),
                "property_type": row.get("property_type", ""),
                "condition": row.get("condition", ""),
                "listing_type": row.get("listing_type", ""),
                "has_outdoor_space": row.get("has_outdoor_space"),
                "image_folder": row.get("image_folder", ""),
                "image_count": row.get("image_count", 0),
                "preview_image": preview_image,
                "image_paths": image_paths,
            }
        )

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"Saved metadata to: {OUTPUT_PATH}")
    print(f"Total listings: {len(merged)}")


if __name__ == "__main__":
    main()