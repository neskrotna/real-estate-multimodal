from __future__ import annotations

import argparse
import numpy as np

from src.utils.io import read_table, write_json
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--train", default=0.70, type=float)
    parser.add_argument("--val", default=0.15, type=float)
    parser.add_argument("--test", default=0.15, type=float)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    if not np.isclose(args.train + args.val + args.test, 1.0):
        raise ValueError("train+val+test must sum to 1.0")

    set_seed(args.seed)

    df = read_table(args.manifest)
    listing_ids = df["listing_id"].dropna().astype(str).unique().tolist()

    rng = np.random.default_rng(args.seed)
    rng.shuffle(listing_ids)

    n = len(listing_ids)
    n_train = int(n * args.train)
    n_val = int(n * args.val)

    splits = {
        "train": listing_ids[:n_train],
        "val": listing_ids[n_train:n_train + n_val],
        "test": listing_ids[n_train + n_val:],
    }

    write_json(args.output, splits)
    print(f"Wrote splits: {args.output}")
    print(f"Train: {len(splits['train'])} | Val: {len(splits['val'])} | Test: {len(splits['test'])} | Total: {n}")


if __name__ == "__main__":
    main()