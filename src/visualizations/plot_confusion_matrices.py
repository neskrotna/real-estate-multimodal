from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from src.utils.io import read_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot confusion matrices for experiments.")
    parser.add_argument("--exp1-metrics", type=Path, default=Path("runs/exp1_clip_maxpool/test_metrics.json"))
    parser.add_argument("--exp2-metrics", type=Path, default=Path("runs/exp2_clip_metadata_classifier/test_metrics.json"))
    parser.add_argument("--exp3-metrics", type=Path, default=Path("runs/exp3_clip_projection_finetune/test_metrics.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/figures"))
    return parser.parse_args()


def save_confusion_matrix(metrics_path: Path, output_path: Path, title: str) -> None:
    metrics = read_json(metrics_path)
    tp, fp, tn, fn = metrics["tp"], metrics["fp"], metrics["tn"], metrics["fn"]

    matrix = [[tn, fp], [fn, tp]]

    plt.figure(figsize=(5, 4))
    plt.imshow(matrix)
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])
    plt.title(title)

    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(matrix[i][j]), ha="center", va="center")

    plt.colorbar()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    save_confusion_matrix(args.exp1_metrics, args.output_dir / "exp1_confusion_matrix.png", "Exp1 MaxPool")
    save_confusion_matrix(args.exp2_metrics, args.output_dir / "exp2_confusion_matrix.png", "Exp2 MetaClassifier")
    save_confusion_matrix(args.exp3_metrics, args.output_dir / "exp3_confusion_matrix.png", "Exp3 FineTune")

    print(f"[INFO] Saved confusion matrices to {args.output_dir}")


if __name__ == "__main__":
    main()