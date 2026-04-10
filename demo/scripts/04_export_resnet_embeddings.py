from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]

METADATA_PATH = PROJECT_ROOT / "demo" / "artifacts" / "listings_metadata.json"
OUTPUT_DIR = PROJECT_ROOT / "demo" / "artifacts" / "resnet"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_model() -> torch.nn.Module:
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model.eval()
    model.to(DEVICE)
    return model


def build_transform():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def load_image_embedding(model, transform, image_path: Path) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        emb = model(tensor).cpu().numpy()[0]

    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb.astype(np.float32)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with METADATA_PATH.open("r", encoding="utf-8") as f:
        listings = json.load(f)

    model = build_model()
    transform = build_transform()

    listing_ids = []
    embeddings = []

    for item in tqdm(listings, desc="Encoding listings with ResNet"):
        preview_image = item.get("preview_image")
        if not preview_image:
            continue

        image_path = PROJECT_ROOT / preview_image
        if not image_path.exists():
            continue

        emb = load_image_embedding(model, transform, image_path)
        listing_ids.append(item["listing_id"])
        embeddings.append(emb)

    embeddings = np.stack(embeddings)
    np.save(OUTPUT_DIR / "embeddings.npy", embeddings)

    with (OUTPUT_DIR / "listing_ids.json").open("w", encoding="utf-8") as f:
        json.dump(listing_ids, f, ensure_ascii=False, indent=2)

    print(f"Saved ResNet embeddings to: {OUTPUT_DIR}")
    print(f"Embeddings shape: {embeddings.shape}")


if __name__ == "__main__":
    main()