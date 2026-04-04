from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

from src.utils.io import read_json, read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build positive, random negative, and hard negative text-image pairs."
    )
    parser.add_argument(
        "--listings",
        type=Path,
        default=Path("data/processed/listings.jsonl"),
        help="Path to listings JSONL",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/manifest.jsonl"),
        help="Path to manifest JSONL",
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=Path("data/processed/split_v1.json"),
        help="Path to split JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory where pair files will be saved",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--random-negatives-per-listing",
        type=int,
        default=2,
        help="How many random negative pairs to create per listing",
    )
    parser.add_argument(
        "--hard-negatives-per-listing",
        type=int,
        default=1,
        help="How many hard negative pairs to create per listing",
    )
    return parser.parse_args()


def build_lookup(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {record["listing_id"]: record for record in records}


def simple_similarity_score(a: dict[str, Any], b: dict[str, Any]) -> float:
    score = 0.0

    if a.get("city") and b.get("city") and a["city"] == b["city"]:
        score += 2.0

    if a.get("listing_type") and b.get("listing_type") and a["listing_type"] == b["listing_type"]:
        score += 1.5

    if a.get("property_type") and b.get("property_type") and a["property_type"] == b["property_type"]:
        score += 1.0

    if a.get("condition") and b.get("condition") and a["condition"] == b["condition"]:
        score += 0.5

    rooms_a = a.get("rooms")
    rooms_b = b.get("rooms")
    if rooms_a is not None and rooms_b is not None:
        diff = abs(float(rooms_a) - float(rooms_b))
        if diff == 0:
            score += 2.0
        elif diff <= 1:
            score += 1.0

    area_a = a.get("area_m2")
    area_b = b.get("area_m2")
    if area_a is not None and area_b is not None:
        diff = abs(float(area_a) - float(area_b))
        if diff <= 10:
            score += 1.5
        elif diff <= 20:
            score += 0.75

    price_a = a.get("price_eur")
    price_b = b.get("price_eur")
    if price_a is not None and price_b is not None:
        diff = abs(float(price_a) - float(price_b))
        if diff <= 250:
            score += 1.5
        elif diff <= 500:
            score += 0.75

    if a.get("has_outdoor_space") is not None and b.get("has_outdoor_space") is not None:
        if a["has_outdoor_space"] == b["has_outdoor_space"]:
            score += 0.5

    return score


def build_pairs_for_split(
    split_name: str,
    split_ids: list[str],
    listings_by_id: dict[str, dict[str, Any]],
    manifest_by_id: dict[str, dict[str, Any]],
    random_negatives_per_listing: int,
    hard_negatives_per_listing: int,
) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []

    split_records = [listings_by_id[lid] for lid in split_ids]

    for listing_id in split_ids:
        listing = listings_by_id[listing_id]
        manifest = manifest_by_id[listing_id]

        text = listing.get("description") or listing.get("title") or ""

        # Positive pair
        pairs.append({
            "pair_type": "positive",
            "label": 1,
            "split": split_name,
            "text_listing_id": listing_id,
            "image_listing_id": listing_id,
            "text": text,
            "image_paths": manifest["image_paths"],
        })

        # Random negatives
        candidate_ids = [lid for lid in split_ids if lid != listing_id]
        random.shuffle(candidate_ids)

        chosen_random_negatives = candidate_ids[:random_negatives_per_listing]
        for neg_id in chosen_random_negatives:
            neg_manifest = manifest_by_id[neg_id]
            pairs.append({
                "pair_type": "random_negative",
                "label": 0,
                "split": split_name,
                "text_listing_id": listing_id,
                "image_listing_id": neg_id,
                "text": text,
                "image_paths": neg_manifest["image_paths"],
            })

        # Hard negatives
        scored_candidates = []
        used_negative_ids = set(chosen_random_negatives)

        for other in split_records:
            other_id = other["listing_id"]
            if other_id == listing_id:
                continue
            if other_id in used_negative_ids:
                continue

            score = simple_similarity_score(listing, other)
            scored_candidates.append((score, other_id))

        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        hard_added = 0
        for score, neg_id in scored_candidates:
            if hard_added >= hard_negatives_per_listing:
                break
            if score <= 0:
                continue

            neg_manifest = manifest_by_id[neg_id]
            pairs.append({
                "pair_type": "hard_negative",
                "label": 0,
                "split": split_name,
                "text_listing_id": listing_id,
                "image_listing_id": neg_id,
                "text": text,
                "image_paths": neg_manifest["image_paths"],
            })
            hard_added += 1

    return pairs


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    listings = read_jsonl(args.listings)
    manifest = read_jsonl(args.manifest)
    split = read_json(args.split)

    listings_by_id = build_lookup(listings)
    manifest_by_id = build_lookup(manifest)

    for split_name in ["train", "val", "test"]:
        split_ids = split[split_name]

        missing_in_listings = [lid for lid in split_ids if lid not in listings_by_id]
        missing_in_manifest = [lid for lid in split_ids if lid not in manifest_by_id]

        if missing_in_listings:
            raise ValueError(f"Missing listing ids in listings.jsonl for split '{split_name}': {missing_in_listings}")
        if missing_in_manifest:
            raise ValueError(f"Missing listing ids in manifest.jsonl for split '{split_name}': {missing_in_manifest}")

        pairs = build_pairs_for_split(
            split_name=split_name,
            split_ids=split_ids,
            listings_by_id=listings_by_id,
            manifest_by_id=manifest_by_id,
            random_negatives_per_listing=args.random_negatives_per_listing,
            hard_negatives_per_listing=args.hard_negatives_per_listing,
        )

        output_path = args.output_dir / f"pairs_{split_name}.jsonl"
        write_jsonl(output_path, pairs)

        print(f"[INFO] Wrote {len(pairs)} pairs to {output_path}")


if __name__ == "__main__":
    main()