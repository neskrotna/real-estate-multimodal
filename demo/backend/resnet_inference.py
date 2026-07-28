from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_model() -> torch.nn.Module:
    """
    Build the same ResNet-50 architecture used when the stored
    listing embeddings were generated.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model.eval()
    model.to(DEVICE)
    return model


def build_transform() -> transforms.Compose:
    """
    Apply the same preprocessing used by 04_export_resnet_embeddings.py.
    """
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


def encode_image(image_path: Path) -> np.ndarray:
    model = build_model()
    transform = build_transform()

    try:
        image = Image.open(image_path).convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError("The uploaded file is not a valid image.") from exc

    tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        embedding = model(tensor).cpu().numpy()[0]

    norm = np.linalg.norm(embedding)

    if norm > 0:
        embedding = embedding / norm

    return embedding.astype(np.float32)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python resnet_inference.py <image_path>"
        )

    image_path = Path(sys.argv[1]).resolve()

    if not image_path.exists():
        raise FileNotFoundError(
            f"Uploaded image was not found: {image_path}"
        )

    embedding = encode_image(image_path)

    # The Node backend reads this JSON array from stdout.
    print(json.dumps(embedding.tolist()))


if __name__ == "__main__":
    main()