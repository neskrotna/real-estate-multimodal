from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from src.utils.io import read_json
from src.visualizations.style import apply_plot_style, finish_plot, PRIMARY_COLOR, SECONDARY_COLOR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training history for a fine-tuning experiment.")
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--title", type=str, default="Fine-tuning loss curves")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    history = read_json(args.history)
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]

    apply_plot_style()
    plt.figure(figsize=(8.5, 5.5))

    plt.plot(epochs, train_loss, label="Train loss", color=PRIMARY_COLOR, linewidth=2.5, marker="o", markersize=4)
    plt.plot(epochs, val_loss, label="Val loss", color=SECONDARY_COLOR, linewidth=2.5, marker="o", markersize=4)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(args.title)
    plt.legend(frameon=True)

    finish_plot(args.output)
    print(f"[INFO] Saved figure to {args.output}")

if __name__ == "__main__":
    main()