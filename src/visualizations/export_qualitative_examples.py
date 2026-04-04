from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.io import read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export qualitative examples from scored predictions.")
    parser.add_argument("--scored-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/examples"))
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rows = read_jsonl(args.scored_file)

    true_positives = [r for r in rows if r["label"] == 1]
    true_negatives = [r for r in rows if r["label"] == 0]

    true_positives_sorted = sorted(true_positives, key=lambda x: x["similarity"], reverse=True)
    true_negatives_sorted = sorted(true_negatives, key=lambda x: x["similarity"])

    hardest_false_positives = sorted(
        [r for r in rows if r["label"] == 0],
        key=lambda x: x["similarity"],
        reverse=True,
    )
    hardest_false_negatives = sorted(
        [r for r in rows if r["label"] == 1],
        key=lambda x: x["similarity"],
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    write_json(args.output_dir / "best_positive_examples.json", true_positives_sorted[: args.top_k])
    write_json(args.output_dir / "best_negative_examples.json", true_negatives_sorted[: args.top_k])
    write_json(args.output_dir / "hardest_false_positive_candidates.json", hardest_false_positives[: args.top_k])
    write_json(args.output_dir / "hardest_false_negative_candidates.json", hardest_false_negatives[: args.top_k])

    print(f"[INFO] Exported qualitative example files to {args.output_dir}")


if __name__ == "__main__":
    main()