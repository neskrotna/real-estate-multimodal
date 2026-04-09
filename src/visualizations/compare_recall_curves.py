from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from src.visualizations.style import (
    SERIES_COLORS,
    apply_plot_style,
    finish_plot,
)

RUNS_DIR = Path("runs")
OUTPUT_DIR = Path("reports/figures")

MODEL_ORDER = [
    "text_tfidf_title_to_description",
    "text_sbert_title_to_description",
    "image_resnet_half_to_half",
    "clip_baseline",
    "exp1_clip_maxpool",
    "exp2_clip_metadata_classifier",
    "exp3_clip_projection_finetune",
    "exp4_clip_only_classifier",
    "exp5_clip_projection_maxpool",
    "exp6_clip_projection_weighted_pooling",
    "exp7_clip_hard_negative_finetune",
    "exp8_clip_pair_classifier",
]

DISPLAY_NAMES = {
    "text_tfidf_title_to_description": "TF-IDF",
    "text_sbert_title_to_description": "SBERT",
    "image_resnet_half_to_half": "ResNet",
    "clip_baseline": "CLIP baseline",
    "exp1_clip_maxpool": "Exp1: CLIP maxpool",
    "exp2_clip_metadata_classifier": "Exp2: metadata classifier",
    "exp3_clip_projection_finetune": "Exp3: projection fine-tuning",
    "exp4_clip_only_classifier": "Exp4: CLIP-only classifier",
    "exp5_clip_projection_maxpool": "Exp5: projection + maxpool",
    "exp6_clip_projection_weighted_pooling": "Exp6: projection + weighted pooling",
    "exp7_clip_hard_negative_finetune": "Exp7: hard negative fine-tuning",
    "exp8_clip_pair_classifier": "Exp8: pair classifier",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_metric(metrics: dict[str, Any], candidates: list[str]) -> float | None:
    normalized = {str(k).lower().strip(): v for k, v in metrics.items()}
    for key in candidates:
        if key.lower() in normalized:
            value = normalized[key.lower()]
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    apply_plot_style()
    fig, ax = plt.subplots(figsize=(12, 7))

    plotted_any = False

    for idx, model_name in enumerate(MODEL_ORDER):
        model_dir = RUNS_DIR / model_name
        if not model_dir.exists():
            continue

        metrics_path = model_dir / "test_metrics.json"
        if not metrics_path.exists():
            metrics_path = model_dir / "metrics.json"
        if not metrics_path.exists():
            continue

        metrics = load_json(metrics_path)

        r1 = get_metric(metrics, ["recall@1", "r@1", "recall_at_1"])
        r5 = get_metric(metrics, ["recall@5", "r@5", "recall_at_5"])
        r10 = get_metric(metrics, ["recall@10", "r@10", "recall_at_10"])

        if r1 is None or r5 is None or r10 is None:
            continue

        xs = [1, 5, 10]
        ys = [r1, r5, r10]

        ax.plot(
            xs,
            ys,
            marker="o",
            linewidth=2.2,
            color=SERIES_COLORS[idx % len(SERIES_COLORS)],
            label=DISPLAY_NAMES.get(model_name, model_name),
        )
        plotted_any = True

    if not plotted_any:
        print("[ERROR] No retrieval models with Recall@K found.")
        return

    ax.set_title("Recall@K comparison across all retrieval models")
    ax.set_xlabel("K")
    ax.set_ylabel("Recall@K")
    ax.set_xticks([1, 5, 10])
    ax.set_ylim(0, 1.05)
    ax.legend(loc="best")

    finish_plot(OUTPUT_DIR / "compare_recall_curves.png")
    print(f"[OK] Saved plot to: {OUTPUT_DIR / 'compare_recall_curves.png'}")


if __name__ == "__main__":
    main()