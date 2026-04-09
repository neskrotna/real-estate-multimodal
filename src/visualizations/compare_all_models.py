from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from src.visualizations.style import (
    PRIMARY_COLOR,
    SECONDARY_COLOR,
    ACCENT_COLOR,
    SOFT_PURPLE,
    DARK_PINK,
    SERIES_COLORS,
    apply_plot_style,
    finish_plot,
)

RUNS_DIR = Path("runs")
OUTPUT_DIR = Path("reports/figures")
OUTPUT_CSV = OUTPUT_DIR / "all_model_metrics_summary.csv"

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


def get_first_metric(metrics: dict[str, Any], candidates: list[str]) -> Any:
    normalized = {str(k).lower().strip(): v for k, v in metrics.items()}
    for key in candidates:
        if key.lower() in normalized:
            return normalized[key.lower()]
    return None


def infer_model_type(model_name: str) -> str:
    if model_name.startswith("text_"):
        return "text"
    if model_name.startswith("image_"):
        return "image"
    if "clip" in model_name or model_name.startswith("exp"):
        return "multimodal"
    return "other"


def collect_metrics() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for model_name in MODEL_ORDER:
        model_dir = RUNS_DIR / model_name
        if not model_dir.exists():
            print(f"[WARN] Missing run directory: {model_dir}")
            continue

        metrics_path = model_dir / "test_metrics.json"
        if not metrics_path.exists():
            metrics_path = model_dir / "metrics.json"

        if not metrics_path.exists():
            print(f"[WARN] No metrics file found for: {model_name}")
            continue

        metrics = load_json(metrics_path)

        row = {
            "model_key": model_name,
            "model": DISPLAY_NAMES.get(model_name, model_name),
            "type": infer_model_type(model_name),
            "recall@1": get_first_metric(metrics, ["recall@1", "r@1", "recall_at_1"]),
            "recall@5": get_first_metric(metrics, ["recall@5", "r@5", "recall_at_5"]),
            "recall@10": get_first_metric(metrics, ["recall@10", "r@10", "recall_at_10"]),
            "mrr": get_first_metric(metrics, ["mrr", "mean_reciprocal_rank"]),
            "accuracy": get_first_metric(metrics, ["accuracy", "acc"]),
            "precision": get_first_metric(metrics, ["precision"]),
            "recall_cls": get_first_metric(metrics, ["recall"]),
            "f1": get_first_metric(metrics, ["f1", "f1_score"]),
            "auc": get_first_metric(metrics, ["auc", "roc_auc"]),
            "metrics_file": str(metrics_path),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def plot_metric_bar(
    df: pd.DataFrame,
    metric_col: str,
    title: str,
    filename: str,
) -> None:
    plot_df = df.dropna(subset=[metric_col]).copy()
    if plot_df.empty:
        print(f"[INFO] Skipping {metric_col}: no values found.")
        return

    plot_df = plot_df.sort_values(by=metric_col, ascending=False)

    apply_plot_style()
    fig, ax = plt.subplots(figsize=(13, 6))

    colors = [SERIES_COLORS[i % len(SERIES_COLORS)] for i in range(len(plot_df))]
    bars = ax.bar(plot_df["model"], plot_df[metric_col], color=colors)

    ax.set_title(title)
    ax.set_ylabel(metric_col)
    ax.set_xlabel("Model")
    ax.set_ylim(0, max(1.0, float(plot_df[metric_col].max()) * 1.15))
    plt.xticks(rotation=35, ha="right")

    for bar, value in zip(bars, plot_df[metric_col]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    finish_plot(OUTPUT_DIR / filename)


def plot_main_retrieval_summary(df: pd.DataFrame) -> None:
    retrieval_df = df.dropna(subset=["recall@1", "recall@5", "recall@10", "mrr"]).copy()
    if retrieval_df.empty:
        print("[INFO] No retrieval metrics found for summary plot.")
        return

    retrieval_df = retrieval_df.set_index("model")[["recall@1", "recall@5", "recall@10", "mrr"]]

    apply_plot_style()
    fig, ax = plt.subplots(figsize=(14, 7))

    retrieval_df.plot(
        kind="bar",
        ax=ax,
        color=[PRIMARY_COLOR, SECONDARY_COLOR, ACCENT_COLOR, SOFT_PURPLE],
        width=0.8,
    )

    ax.set_title("All model comparison across retrieval metrics")
    ax.set_ylabel("Score")
    ax.set_xlabel("Model")
    ax.set_ylim(0, 1.1)
    plt.xticks(rotation=35, ha="right")
    plt.legend(title="Metric")

    finish_plot(OUTPUT_DIR / "full_model_comparison.png")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = collect_metrics()
    if df.empty:
        print("[ERROR] No metrics found.")
        return

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[OK] Saved summary CSV to: {OUTPUT_CSV}")

    plot_main_retrieval_summary(df)
    plot_metric_bar(df, "recall@1", "Recall@1 comparison across all models", "compare_recall_at_1.png")
    plot_metric_bar(df, "recall@5", "Recall@5 comparison across all models", "compare_recall_at_5.png")
    plot_metric_bar(df, "recall@10", "Recall@10 comparison across all models", "compare_recall_at_10.png")
    plot_metric_bar(df, "mrr", "MRR comparison across all models", "compare_mrr.png")
    plot_metric_bar(df, "accuracy", "Accuracy comparison across all models", "compare_accuracy.png")
    plot_metric_bar(df, "f1", "F1 comparison across all models", "compare_f1.png")


if __name__ == "__main__":
    main()