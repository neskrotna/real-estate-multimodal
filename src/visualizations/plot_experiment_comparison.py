from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.utils.io import read_json
from src.visualizations.style import apply_plot_style, finish_plot, SERIES_COLORS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot side-by-side experiment metric comparison.")
    parser.add_argument(
        "--metrics-files",
        nargs="+",
        required=True,
        help="List of metric JSON files",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        required=True,
        help="List of labels matching the metric files",
    )
    parser.add_argument("--output", type=Path, default=Path("reports/figures/experiment_comparison.png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if len(args.metrics_files) != len(args.labels):
        raise ValueError("--metrics-files and --labels must have the same length")

    apply_plot_style()

    metric_names = ["accuracy", "precision", "recall", "f1"]
    pretty_names = ["Accuracy", "Precision", "Recall", "F1 Score"]
    all_metrics = [read_json(Path(p)) for p in args.metrics_files]

    x = np.arange(len(metric_names))
    n = len(all_metrics)
    width = 0.8 / n

    plt.figure(figsize=(12, 6))

    for i, (label, metrics) in enumerate(zip(args.labels, all_metrics)):
        values = [metrics[m] for m in metric_names]
        offset = (i - (n - 1) / 2) * width
        color = SERIES_COLORS[i % len(SERIES_COLORS)]
        bars = plt.bar(x + offset, values, width=width, label=label, color=color, edgecolor="white", linewidth=1.0)

        for bar, value in zip(bars, values):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.015,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.xticks(x, pretty_names)
    plt.ylim(0, 1.08)
    plt.ylabel("Score")
    plt.title("Experiment Comparison on Test Set")
    plt.legend(frameon=True)

    finish_plot(args.output)
    print(f"[INFO] Saved figure to {args.output}")

if __name__ == "__main__":
    main()