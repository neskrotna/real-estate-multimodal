from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.io import read_json, read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export qualitative TP/TN/FP/FN examples from scored predictions."
    )
    parser.add_argument("--scored-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/examples"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--metrics-file", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rows = read_jsonl(args.scored_file)

    if args.threshold is None:
        if args.metrics_file is None:
            raise ValueError("Provide either --threshold or --metrics-file")
        metrics = read_json(args.metrics_file)
        threshold = float(metrics["threshold"])
    else:
        threshold = float(args.threshold)

    for row in rows:
        score = row.get("match_probability", row.get("similarity"))
        pred = 1 if score >= threshold else 0
        row["predicted_label"] = pred

    tp = [r for r in rows if r["label"] == 1 and r["predicted_label"] == 1]
    tn = [r for r in rows if r["label"] == 0 and r["predicted_label"] == 0]
    fp = [r for r in rows if r["label"] == 0 and r["predicted_label"] == 1]
    fn = [r for r in rows if r["label"] == 1 and r["predicted_label"] == 0]

    tp = sorted(tp, key=lambda x: x.get("match_probability", x.get("similarity", 0.0)), reverse=True)
    tn = sorted(tn, key=lambda x: x.get("match_probability", x.get("similarity", 0.0)))
    fp = sorted(fp, key=lambda x: x.get("match_probability", x.get("similarity", 0.0)), reverse=True)
    fn = sorted(fn, key=lambda x: x.get("match_probability", x.get("similarity", 0.0)))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    write_json(args.output_dir / "true_positives.json", tp[: args.top_k])
    write_json(args.output_dir / "true_negatives.json", tn[: args.top_k])
    write_json(args.output_dir / "false_positives.json", fp[: args.top_k])
    write_json(args.output_dir / "false_negatives.json", fn[: args.top_k])

    print(f"[INFO] Exported qualitative examples to {args.output_dir}")

if __name__ == "__main__":
    main()