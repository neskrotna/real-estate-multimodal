from __future__ import annotations

import json
import pickle
from pathlib import Path

from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]

METADATA_PATH = PROJECT_ROOT / "demo" / "artifacts" / "listings_metadata.json"
OUTPUT_DIR = PROJECT_ROOT / "demo" / "artifacts" / "tfidf"


def build_corpus(item: dict) -> str:
    parts = [
        item.get("title", ""),
        item.get("description", ""),
        item.get("city", ""),
        str(item.get("rooms", "") or ""),
        str(item.get("area_m2", "") or ""),
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
    corpus = [build_corpus(item) for item in listings]

    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        lowercase=True,
        strip_accents=None,
    )

    matrix = vectorizer.fit_transform(corpus)

    with (OUTPUT_DIR / "vectorizer.pkl").open("wb") as f:
        pickle.dump(vectorizer, f)

    sparse.save_npz(OUTPUT_DIR / "matrix.npz", matrix)

    with (OUTPUT_DIR / "listing_ids.json").open("w", encoding="utf-8") as f:
        json.dump(listing_ids, f, ensure_ascii=False, indent=2)

    print(f"Saved TF-IDF artifacts to: {OUTPUT_DIR}")
    print(f"Matrix shape: {matrix.shape}")


if __name__ == "__main__":
    main()