from __future__ import annotations

import argparse
from email import parser
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc

from src.utils.io import read_jsonl
from src.visualizations.style import apply_plot_style, finish_plot, SERIES_COLORS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot precision-recall curves for multiple experiments.")
    parser.add_argument(
        "--scored-files",
        nargs="+",
        required=True,
        help="List of scored JSONL files",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        required=True,
        help="Labels matching the scored files",
    )
    parser.add_argument("--output", type=Path, default=Path("reports/figures/precision_recall_curves.png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if len(args.scored_files) != len(args.labels):
        raise ValueError("--scored-files and --labels must have the same length")

    apply_plot_style()
    plt.figure(figsize=(8.5, 6))

    for i, (scored_file, label) in enumerate(zip(args.scored_files, args.labels)):
        rows = read_jsonl(Path(scored_file))
        y_true = [r["label"] for r in rows]
        y_score = [r.get("match_probability", r.get("similarity")) for r in rows]

        precision, recall, _ = precision_recall_curve(y_true, y_score)
        pr_auc = auc(recall, precision)

        plt.plot(recall, precision, label=f"{label} (AUC={pr_auc:.3f})", color=SERIES_COLORS[i % len(SERIES_COLORS)], linewidth=2)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend(frameon=True)

    finish_plot(args.output)
    print(f"[INFO] Saved figure to {args.output}")

if __name__ == "__main__":
    main()