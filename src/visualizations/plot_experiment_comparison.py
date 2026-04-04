from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from src.utils.io import read_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot side-by-side experiment metric comparison.")
    parser.add_argument("--exp1-metrics", type=Path, default=Path("runs/exp1_clip_maxpool/test_metrics.json"))
    parser.add_argument("--exp2-metrics", type=Path, default=Path("runs/exp2_clip_metadata_classifier/test_metrics.json"))
    parser.add_argument("--exp3-metrics", type=Path, default=Path("runs/exp3_clip_projection_finetune/test_metrics.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/figures/experiment_comparison.png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    exp1 = read_json(args.exp1_metrics)
    exp3 = read_json(args.exp3_metrics)
    exp4 = read_json(args.exp4_metrics)

    labels = ["Exp1 MaxPool", "Exp2 MetaClassifier", "Exp3 FineTune"]
    metrics = ["accuracy", "precision", "recall", "f1"]
    values = [
        [exp1[m] for m in metrics],
        [exp3[m] for m in metrics],
        [exp4[m] for m in metrics],
    ]

    x = range(len(metrics))
    width = 0.25

    plt.figure(figsize=(10, 6))
    plt.bar([i - width for i in x], values[0], width=width, label=labels[0])
    plt.bar(list(x), values[1], width=width, label=labels[1])
    plt.bar([i + width for i in x], values[2], width=width, label=labels[2])

    plt.xticks(list(x), metrics)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Experiment comparison on test set")
    plt.legend()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.output, dpi=200)
    plt.close()

    print(f"[INFO] Saved figure to {args.output}")


if __name__ == "__main__":
    main()