from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from src.utils.io import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot similarity distributions for scored pairs.")
    parser.add_argument("--scored-file", type=Path, required=True)
    parser.add_argument("--title", type=str, default="Similarity distributions")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rows = read_jsonl(args.scored_file)
    pos = [r["similarity"] for r in rows if r["label"] == 1]
    neg = [r["similarity"] for r in rows if r["label"] == 0]

    plt.figure(figsize=(8, 5))
    plt.hist(pos, bins=20, alpha=0.6, label="Positive")
    plt.hist(neg, bins=20, alpha=0.6, label="Negative")
    plt.xlabel("Similarity")
    plt.ylabel("Count")
    plt.title(args.title)
    plt.legend()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.output, dpi=200)
    plt.close()

    print(f"[INFO] Saved figure to {args.output}")


if __name__ == "__main__":
    main()