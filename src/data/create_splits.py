from __future__ import annotations

import argparse
import random
from pathlib import Path

from src.utils.io import read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create train/val/test splits on listing level."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/listings.jsonl"),
        help="Path to listings JSONL",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/split_v1.json"),
        help="Output split JSON file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.6,
        help="Train split ratio",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation split ratio",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Test split ratio",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-8:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    listings = read_jsonl(args.input)
    listing_ids = [item["listing_id"] for item in listings]

    random.seed(args.seed)
    random.shuffle(listing_ids)

    n_total = len(listing_ids)
    n_train = int(n_total * args.train_ratio)
    n_val = int(n_total * args.val_ratio)
    n_test = n_total - n_train - n_val

    train_ids = listing_ids[:n_train]
    val_ids = listing_ids[n_train:n_train + n_val]
    test_ids = listing_ids[n_train + n_val:]

    split = {
        "seed": args.seed,
        "ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "counts": {
            "total": n_total,
            "train": len(train_ids),
            "val": len(val_ids),
            "test": len(test_ids),
        },
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
    }

    write_json(args.output, split)

    print("=== Split Summary ===")
    print(f"Total listings: {n_total}")
    print(f"Train: {len(train_ids)}")
    print(f"Val:   {len(val_ids)}")
    print(f"Test:  {len(test_ids)}")
    print(f"[INFO] Split written to: {args.output}")


if __name__ == "__main__":
    main()