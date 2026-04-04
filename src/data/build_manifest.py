from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.utils.io import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build image manifest from listings and image folders."
    )
    parser.add_argument(
        "--listings",
        type=Path,
        default=Path("data/processed/listings.jsonl"),
        help="Path to listings JSONL",
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        default=Path("data/processed/images"),
        help="Root directory of image folders",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/manifest.jsonl"),
        help="Output manifest JSONL",
    )
    return parser.parse_args()


def get_image_paths(folder: Path) -> list[str]:
    if not folder.exists():
        return []

    image_files = list(folder.glob("*.jpg"))

    # Sort to ensure consistent order
    image_files = sorted(image_files)

    return [str(p.as_posix()) for p in image_files]


def main() -> None:
    args = parse_args()

    listings = read_jsonl(args.listings)

    manifest_records: list[dict[str, Any]] = []

    missing_folders = 0
    empty_folders = 0
    count_mismatch = 0

    for listing in listings:
        listing_id = listing["listing_id"]
        folder_name = listing["image_folder"]

        folder_path = args.images_root / folder_name

        if not folder_path.exists():
            missing_folders += 1
            image_paths = []
        else:
            image_paths = get_image_paths(folder_path)

        if len(image_paths) == 0:
            empty_folders += 1

        # Compare with expected count from Excel
        expected_count = listing.get("image_count")
        if expected_count is not None and expected_count != len(image_paths):
            count_mismatch += 1
            print(f"[WARN] {listing_id}: excel={expected_count}, actual={len(image_paths)}")

        record = {
            "listing_id": listing_id,
            "image_folder": folder_name,
            "image_paths": image_paths,
            "num_images": len(image_paths),
        }

        manifest_records.append(record)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, manifest_records)

    print("=== Manifest Summary ===")
    print(f"Total listings: {len(listings)}")
    print(f"Missing folders: {missing_folders}")
    print(f"Empty folders: {empty_folders}")
    print(f"Count mismatches: {count_mismatch}")
    print(f"[INFO] Manifest written to: {args.output}")


if __name__ == "__main__":
    main()