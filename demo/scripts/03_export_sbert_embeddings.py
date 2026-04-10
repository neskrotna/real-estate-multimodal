from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]

METADATA_PATH = PROJECT_ROOT / "demo" / "artifacts" / "listings_metadata.json"
OUTPUT_DIR = PROJECT_ROOT / "demo" / "artifacts" / "sbert"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


def build_text(item: dict) -> str:
    parts = [
        item.get("title", ""),
        item.get("description", ""),
        item.get("city", ""),
        f"rooms {item.get('rooms', '')}",
        f"area {item.get('area_m2', '')}",
        item.get("property_type", ""),
        item.get("condition", ""),
        item.get("listing_type", ""),
    ]
    return " ".join(part for part in parts if part).strip()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with METADATA_PATH.open("r", encoding="utf-8") as f:
        listings = json.load(f)

    listing_ids = [item["listing_id"] for item in listings]
    texts = [build_text(item) for item in listings]

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    np.save(OUTPUT_DIR / "embeddings.npy", embeddings)

    with (OUTPUT_DIR / "listing_ids.json").open("w", encoding="utf-8") as f:
        json.dump(listing_ids, f, ensure_ascii=False, indent=2)

    print(f"Saved SBERT embeddings to: {OUTPUT_DIR}")
    print(f"Embeddings shape: {embeddings.shape}")


if __name__ == "__main__":
    main()