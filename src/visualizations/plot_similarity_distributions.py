from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from src.utils.io import read_json, read_jsonl
from src.visualizations.style import apply_plot_style, finish_plot, PRIMARY_COLOR, SECONDARY_COLOR, DARK_PINK


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot score distributions for scored pairs.")
    parser.add_argument("--scored-file", type=Path, required=True)
    parser.add_argument("--title", type=str, default="Score distributions")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics-file", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rows = read_jsonl(args.scored_file)
    scores = []
    labels = []

    for r in rows:
        score = r.get("match_probability", r.get("similarity"))
        scores.append(score)
        labels.append(r["label"])

    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]

    apply_plot_style()
    plt.figure(figsize=(8.5, 5.5))

    plt.hist(pos, bins=20, alpha=0.7, label="Positive", color=PRIMARY_COLOR, edgecolor="white")
    plt.hist(neg, bins=20, alpha=0.7, label="Negative", color=SECONDARY_COLOR, edgecolor="white")

    if args.metrics_file is not None:
        metrics = read_json(args.metrics_file)
        threshold = metrics["threshold"]
        plt.axvline(threshold, linestyle="--", linewidth=2, color=DARK_PINK, label=f"Threshold = {threshold:.2f}")

    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.title(args.title)
    plt.legend(frameon=True)

    finish_plot(args.output)
    print(f"[INFO] Saved figure to {args.output}")

if __name__ == "__main__":
    main()