from __future__ import annotations

import json
from pathlib import Path
from typing import Any

RUNS_DIR = Path("runs")
OUTPUT_DIR = Path("reports/examples")
OUTPUT_FILE = OUTPUT_DIR / "cross_model_topk_comparison.md"

MODEL_ORDER = [
    "text_tfidf_title_to_description",
    "text_sbert_title_to_description",
    "image_resnet_half_to_half",
]

DISPLAY_NAMES = {
    "text_tfidf_title_to_description": "TF-IDF",
    "text_sbert_title_to_description": "SBERT",
    "image_resnet_half_to_half": "ResNet",
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def truncate(text: str, max_len: int = 160) -> str:
    text = str(text).strip().replace("\n", " ")
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def format_prediction(pred: dict[str, Any], rank: int) -> str:
    parts = [f"**Rank {rank}**"]

    for key in ["candidate_id", "listing_id", "match_id", "id"]:
        if key in pred:
            parts.append(f"- Candidate: `{pred[key]}`")
            break

    for key in ["score", "similarity", "probability", "match_score"]:
        if key in pred:
            parts.append(f"- Score: `{pred[key]}`")
            break

    for key in ["title", "candidate_title", "description", "candidate_description", "text"]:
        if key in pred:
            parts.append(f"- Text: {truncate(str(pred[key]))}")
            break

    for key in ["image_path", "candidate_image_path", "path"]:
        if key in pred:
            parts.append(f"- Image: `{pred[key]}`")
            break

    return "\n".join(parts)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    content: list[str] = []
    content.append("# Cross-model qualitative comparison\n")
    content.append(
        "This file compares top-k predictions across selected baselines for qualitative analysis.\n"
    )

    for model_name in MODEL_ORDER:
        preds_path = RUNS_DIR / model_name / "top5_predictions.jsonl"
        if not preds_path.exists():
            content.append(f"## {DISPLAY_NAMES.get(model_name, model_name)}\n")
            content.append(f"Missing file: `{preds_path}`\n")
            continue

        rows = []
        with preds_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))

        content.append(f"## {DISPLAY_NAMES.get(model_name, model_name)}\n")

        for query_idx, row in enumerate(rows[:10], start=1):
            content.append(f"### Query {query_idx}\n")

            query_text = None
            for key in ["query", "query_text", "title", "description", "source_text"]:
                if key in row:
                    query_text = row[key]
                    break

            query_id = None
            for key in ["query_id", "listing_id", "id"]:
                if key in row:
                    query_id = row[key]
                    break

            if query_id is not None:
                content.append(f"- Query ID: `{query_id}`")

            if query_text is not None:
                content.append(f"- Query text: {truncate(str(query_text), 220)}")

            predictions = None
            for key in ["top5", "predictions", "results", "top_k"]:
                if key in row and isinstance(row[key], list):
                    predictions = row[key]
                    break

            if not predictions:
                content.append("- No predictions found.\n")
                continue

            content.append("")
            for rank, pred in enumerate(predictions[:5], start=1):
                content.append(format_prediction(pred, rank))
                content.append("")

    OUTPUT_FILE.write_text("\n".join(content), encoding="utf-8")
    print(f"[OK] Saved markdown file to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()