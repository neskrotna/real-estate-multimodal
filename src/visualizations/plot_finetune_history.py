from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from src.utils.io import read_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training history for fine-tuning experiment.")
    parser.add_argument("--history", type=Path, default=Path("runs/exp3_clip_projection_finetune/history.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/figures/exp3_training_curve.png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    history = read_json(args.history)
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label="Train loss")
    plt.plot(epochs, val_loss, label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Fine-tuning loss curves")
    plt.legend()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.output, dpi=200)
    plt.close()

    print(f"[INFO] Saved figure to {args.output}")


if __name__ == "__main__":
    main()