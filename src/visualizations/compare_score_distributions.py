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
    for c in candidates:
        if c in lower_map:
            return lower_map[c]
    return None


def get_label_key(row: dict[str, Any]) -> str | None:
    candidates = ["label", "target", "y_true", "is_match", "match"]
    lower_map = {str(k).lower(): k for k in row.keys()}
    for c in candidates:
        if c in lower_map:
            return lower_map[c]
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
    fig, ax = plt.subplots(figsize=(13, 7))

    plotted_any = False

    for idx, model_name in enumerate(MODEL_ORDER):
        scored_file = RUNS_DIR / model_name / "scored_test.jsonl"
        if not scored_file.exists():
            print(f"[WARN] Missing scored file: {scored_file}")
            continue

        rows = load_jsonl(scored_file)
        pos_scores, neg_scores = extract_scores(rows)

        if not pos_scores or not neg_scores:
            print(f"[WARN] Could not extract scores/labels for {model_name}")
            continue

        ax.hist(
            pos_scores,
            bins=35,
            alpha=0.35,
            density=True,
            color=SERIES_COLORS[idx % len(SERIES_COLORS)],
            label=f"{DISPLAY_NAMES.get(model_name, model_name)} positives",
        )
        ax.hist(
            neg_scores,
            bins=35,
            alpha=0.18,
            density=True,
            histtype="step",
            linewidth=2,
            color=SERIES_COLORS[idx % len(SERIES_COLORS)],
            label=f"{DISPLAY_NAMES.get(model_name, model_name)} negatives",
        )
        plotted_any = True

    if not plotted_any:
        print("[ERROR] Nothing plotted. Check scored_test.jsonl format.")
        return

    ax.set_title("Score distributions across selected multimodal models")
    ax.set_xlabel("Predicted score / similarity")
    ax.set_ylabel("Density")
    ax.legend(loc="best", fontsize=9)

    finish_plot(OUTPUT_DIR / "compare_score_distributions.png")
    print(f"[OK] Saved plot to: {OUTPUT_DIR / 'compare_score_distributions.png'}")


if __name__ == "__main__":
    main()