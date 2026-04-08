from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from src.utils.io import read_json
from src.visualizations.style import apply_plot_style, finish_plot, TEXT_COLOR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot confusion matrices for multiple experiments.")
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
    parser.add_argument("--output-dir", type=Path, default=Path("reports/figures/confusion_matrices"))
    return parser.parse_args()


def save_confusion_matrix(metrics_path: Path, output_path: Path, title: str) -> None:
    metrics = read_json(metrics_path)
    tp, fp, tn, fn = metrics["tp"], metrics["fp"], metrics["tn"], metrics["fn"]

    matrix = [[tn, fp], [fn, tp]]

    apply_plot_style()
    plt.figure(figsize=(5.5, 4.5))
    im = plt.imshow(matrix, cmap="RdPu")

    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])
    plt.title(title)

    max_value = max(max(row) for row in matrix)
    for i in range(2):
        for j in range(2):
            value = matrix[i][j]
            text_color = "white" if value > max_value / 2 else TEXT_COLOR
            plt.text(j, i, str(value), ha="center", va="center", fontsize=12, fontweight="bold", color=text_color)

    plt.colorbar(im, fraction=0.046, pad=0.04)
    finish_plot(output_path)


def main() -> None:
    args = parse_args()

    if len(args.metrics_files) != len(args.labels):
        raise ValueError("--metrics-files and --labels must have the same length")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for metrics_file, label in zip(args.metrics_files, args.labels):
        metrics_path = Path(metrics_file)
        safe_name = label.lower().replace(" ", "_").replace("/", "_")
        output_path = args.output_dir / f"{safe_name}_confusion_matrix.png"
        save_confusion_matrix(metrics_path, output_path, label)

    print(f"[INFO] Saved confusion matrices to {args.output_dir}")

if __name__ == "__main__":
    main()