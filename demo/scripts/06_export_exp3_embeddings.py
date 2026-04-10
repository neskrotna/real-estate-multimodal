from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]

METADATA_PATH = PROJECT_ROOT / "demo" / "artifacts" / "listings_metadata.json"
OUTPUT_DIR = PROJECT_ROOT / "demo" / "artifacts" / "exp3"

BEST_MODEL_PATH = PROJECT_ROOT / "runs" / "exp3_clip_projection_finetune" / "best_model.pt"

TEXT_MODEL_NAME = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
IMAGE_MODEL_NAME = "clip-ViT-B-32"

EMBEDDING_DIM = 512
PROJECTION_DIM = 256
MAX_IMAGES_PER_LISTING = 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ProjectionModel(nn.Module):
    def __init__(self, embedding_dim: int = 512, projection_dim: int = 256):
        super().__init__()
        self.text_head = nn.Sequential(
            nn.Linear(embedding_dim, projection_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(projection_dim, projection_dim),
        )
        self.image_head = nn.Sequential(
            nn.Linear(embedding_dim, projection_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(projection_dim, projection_dim),
        )

    def forward(self, text_x: torch.Tensor, image_x: torch.Tensor):
        text_z = self.text_head(text_x)
        image_z = self.image_head(image_x)
        text_z = F.normalize(text_z, p=2, dim=-1)
        image_z = F.normalize(image_z, p=2, dim=-1)
        return text_z, image_z


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def normalize_np(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.astype(np.float32)


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

    print(f"[INFO] Using device: {DEVICE}")
    print(f"[INFO] Loading text model: {TEXT_MODEL_NAME}")
    print(f"[INFO] Loading image model: {IMAGE_MODEL_NAME}")

    text_model = SentenceTransformer(TEXT_MODEL_NAME, device=str(DEVICE))
    image_model = SentenceTransformer(IMAGE_MODEL_NAME, device=str(DEVICE))

    print(f"[INFO] Loading projection model from: {BEST_MODEL_PATH}")
    projection_model = ProjectionModel(
        embedding_dim=EMBEDDING_DIM,
        projection_dim=PROJECTION_DIM,
    )
    state_dict = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
    projection_model.load_state_dict(state_dict)
    projection_model.to(DEVICE)
    projection_model.eval()

    listing_ids: list[str] = []
    text_embeddings: list[np.ndarray] = []
    image_embeddings: list[np.ndarray] = []
    combined_embeddings: list[np.ndarray] = []

    for item in tqdm(listings, desc="Exporting Exp3 embeddings"):
        listing_id = item["listing_id"]
        image_paths = item.get("image_paths", [])[:MAX_IMAGES_PER_LISTING]

        if not image_paths:
            continue

        valid_image_paths = []
        for rel_path in image_paths:
            abs_path = PROJECT_ROOT / rel_path
            if abs_path.exists():
                valid_image_paths.append(abs_path)

        if not valid_image_paths:
            continue

        text = build_text(item)

        # frozen text embedding
        text_emb = text_model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

        # frozen image embedding with mean pooling across listing images
        images = [load_image(p) for p in valid_image_paths]
        image_embs = image_model.encode(
            images,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        pooled_image_emb = np.mean(image_embs, axis=0)
        pooled_image_emb = pooled_image_emb / np.linalg.norm(pooled_image_emb)

        with torch.no_grad():
            text_tensor = torch.tensor(text_emb, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            image_tensor = torch.tensor(pooled_image_emb, dtype=torch.float32, device=DEVICE).unsqueeze(0)

            proj_text = projection_model.text_head(text_tensor)
            proj_text = F.normalize(proj_text, p=2, dim=-1).squeeze(0).cpu().numpy()

            proj_image = projection_model.image_head(image_tensor)
            proj_image = F.normalize(proj_image, p=2, dim=-1).squeeze(0).cpu().numpy()

        proj_text = normalize_np(proj_text)
        proj_image = normalize_np(proj_image)
        combined = normalize_np((proj_text + proj_image) / 2.0)

        listing_ids.append(listing_id)
        text_embeddings.append(proj_text)
        image_embeddings.append(proj_image)
        combined_embeddings.append(combined)

    text_embeddings = np.stack(text_embeddings)
    image_embeddings = np.stack(image_embeddings)
    combined_embeddings = np.stack(combined_embeddings)

    np.save(OUTPUT_DIR / "text_embeddings.npy", text_embeddings)
    np.save(OUTPUT_DIR / "image_embeddings.npy", image_embeddings)
    np.save(OUTPUT_DIR / "combined_embeddings.npy", combined_embeddings)

    with (OUTPUT_DIR / "listing_ids.json").open("w", encoding="utf-8") as f:
        json.dump(listing_ids, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Saved Exp3 artifacts to: {OUTPUT_DIR}")
    print(f"[INFO] Text embeddings shape: {text_embeddings.shape}")
    print(f"[INFO] Image embeddings shape: {image_embeddings.shape}")
    print(f"[INFO] Combined embeddings shape: {combined_embeddings.shape}")


if __name__ == "__main__":
    main()