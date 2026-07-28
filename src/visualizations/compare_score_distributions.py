from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from src.visualizations.style import apply_plot_style, finish_plot

RUNS_DIR = Path("runs")
OUTPUT_DIR = Path("reports/figures")

MODEL_ORDER = [
    "clip_baseline",
    "exp3_clip_projection_finetune",
    "exp5_clip_projection_maxpool",
]

DISPLAY_NAMES = {
    "clip_baseline": "CLIP baseline",
    "exp3_clip_projection_finetune": "Exp3: projection fine-tuning",
    "exp5_clip_projection_maxpool": "Exp5: projection + maxpool",
}

POSITIVE_COLORS = {
    "clip_baseline": "#F48FB1",
    "exp3_clip_projection_finetune": "#EC407A",
    "exp5_clip_projection_maxpool": "#CE93D8",
}

NEGATIVE_COLORS = {
    "clip_baseline": "#444444",
    "exp3_clip_projection_finetune": "#222222",
    "exp5_clip_projection_maxpool": "#666666",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def get_score_key(row: dict[str, Any]) -> str | None:
    candidates = ["score", "similarity", "probability", "pred_score", "match_score"]
    lower_map = {str(k).lower(): k for k in row.keys()}
    for candidate in candidates:
        if candidate in lower_map:
            return lower_map[candidate]
    return None


def get_label_key(row: dict[str, Any]) -> str | None:
    candidates = ["label", "target", "y_true", "is_match", "match"]
    lower_map = {str(k).lower(): k for k in row.keys()}
    for candidate in candidates:
        if candidate in lower_map:
            return lower_map[candidate]
    return None


def normalize_label(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)

    if isinstance(value, (int, float)):
        return int(value)

    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "positive", "pos", "match", "matched"}:
            return 1
        if v in {"0", "false", "negative", "neg", "mismatch", "not_match"}:
            return 0

    return None


def extract_scores(rows: list[dict[str, Any]]) -> tuple[list[float], list[float]]:
    if not rows:
        return [], []

    score_key = get_score_key(rows[0])
    label_key = get_label_key(rows[0])

    if score_key is None or label_key is None:
        return [], []

    positives: list[float] = []
    negatives: list[float] = []

    for row in rows:
        score = row.get(score_key)
        label = normalize_label(row.get(label_key))

        try:
            score = float(score)
        except (TypeError, ValueError):
            continue

        if label == 1:
            positives.append(score)
        elif label == 0:
            negatives.append(score)

    return positives, negatives


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    apply_plot_style()

    fig, axes = plt.subplots(
        nrows=len(MODEL_ORDER),
        ncols=1,
        figsize=(11, 10),
        sharex=True,
    )

    if len(MODEL_ORDER) == 1:
        axes = [axes]

    plotted_any = False

    for ax, model_name in zip(axes, MODEL_ORDER):
        scored_file = RUNS_DIR / model_name / "scored_test.jsonl"
        if not scored_file.exists():
            ax.set_visible(False)
            print(f"[WARN] Missing scored file: {scored_file}")
            continue

        rows = load_jsonl(scored_file)
        pos_scores, neg_scores = extract_scores(rows)

        if not pos_scores or not neg_scores:
            ax.set_visible(False)
            print(f"[WARN] Could not extract scores/labels for {model_name}")
            continue

        ax.hist(
            neg_scores,
            bins=25,
            density=True,
            alpha=0.95,
            histtype="step",
            linewidth=2.2,
            linestyle="--",
            color=NEGATIVE_COLORS[model_name],
            label="Negative pairs",
        )

        ax.hist(
            pos_scores,
            bins=25,
            density=True,
            alpha=0.45,
            color=POSITIVE_COLORS[model_name],
            edgecolor=NEGATIVE_COLORS[model_name],
            linewidth=0.6,
            label="Positive pairs",
        )

        ax.set_title(DISPLAY_NAMES[model_name], loc="left", fontsize=13, fontweight="bold")
        ax.set_ylabel("Density")
        ax.legend(loc="upper right", frameon=True)
        plotted_any = True

    if not plotted_any:
        print("[ERROR] Nothing plotted. Check scored_test.jsonl format.")
        return

    axes[-1].set_xlabel("Predicted score / similarity")
    fig.suptitle(
        "Score distributions for selected multimodal models",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    finish_plot(OUTPUT_DIR / "compare_score_distributions_qg2.png")
    print(f"[OK] Saved plot to: {OUTPUT_DIR / 'compare_score_distributions_qg2.png'}")


if __name__ == "__main__":
    main()