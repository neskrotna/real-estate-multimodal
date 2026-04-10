from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


PROJECT_ROOT = Path(__file__).resolve().parents[2]

METADATA_PATH = PROJECT_ROOT / "demo" / "artifacts" / "listings_metadata.json"
OUTPUT_DIR = PROJECT_ROOT / "demo" / "artifacts" / "clip"

MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_text(item: dict) -> str:
    parts = [
        item.get("title", ""),
        item.get("description", ""),
        item.get("city", ""),
        f"rooms {item.get('rooms', '')}",
        f"area {item.get('area_m2', '')}",
        item.get("property_type", ""),
        item.get("condition", ""),
    ]
    return " ".join(part for part in parts if part).strip()


def normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.astype(np.float32)


def extract_tensor(output):
    """
    Make the script robust across different transformers output types.
    """
    if isinstance(output, torch.Tensor):
        return output

    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output

    if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
        # fallback: take CLS token
        return output.last_hidden_state[:, 0, :]

    if isinstance(output, (tuple, list)) and len(output) > 0:
        first = output[0]
        if isinstance(first, torch.Tensor):
            return first

    raise TypeError(f"Could not extract tensor from output type: {type(output)}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with METADATA_PATH.open("r", encoding="utf-8") as f:
        listings = json.load(f)

    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()

    listing_ids = []
    image_embeddings = []
    text_embeddings = []
    combined_embeddings = []

    for item in tqdm(listings, desc="Encoding listings with CLIP"):
        preview_image = item.get("preview_image")
        if not preview_image:
            continue

        image_path = PROJECT_ROOT / preview_image
        if not image_path.exists():
            continue

        text = build_text(item)
        image = Image.open(image_path).convert("RGB")

        image_inputs = processor(images=image, return_tensors="pt")
        image_inputs = {k: v.to(DEVICE) for k, v in image_inputs.items()}

        text_inputs = processor(
            text=[text],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        text_inputs = {k: v.to(DEVICE) for k, v in text_inputs.items()}

        with torch.no_grad():
            raw_image_features = model.get_image_features(**image_inputs)
            raw_text_features = model.get_text_features(**text_inputs)

            image_features = extract_tensor(raw_image_features)
            text_features = extract_tensor(raw_text_features)

        image_vec = image_features.detach().cpu().numpy()[0]
        text_vec = text_features.detach().cpu().numpy()[0]

        image_vec = normalize(image_vec)
        text_vec = normalize(text_vec)
        combined_vec = normalize((image_vec + text_vec) / 2.0)

        listing_ids.append(item["listing_id"])
        image_embeddings.append(image_vec)
        text_embeddings.append(text_vec)
        combined_embeddings.append(combined_vec)

    image_embeddings = np.stack(image_embeddings)
    text_embeddings = np.stack(text_embeddings)
    combined_embeddings = np.stack(combined_embeddings)

    np.save(OUTPUT_DIR / "image_embeddings.npy", image_embeddings)
    np.save(OUTPUT_DIR / "text_embeddings.npy", text_embeddings)
    np.save(OUTPUT_DIR / "combined_embeddings.npy", combined_embeddings)

    with (OUTPUT_DIR / "listing_ids.json").open("w", encoding="utf-8") as f:
        json.dump(listing_ids, f, ensure_ascii=False, indent=2)

    print(f"Saved CLIP embeddings to: {OUTPUT_DIR}")
    print(f"Image embeddings shape: {image_embeddings.shape}")
    print(f"Text embeddings shape: {text_embeddings.shape}")
    print(f"Combined embeddings shape: {combined_embeddings.shape}")


if __name__ == "__main__":
    main()