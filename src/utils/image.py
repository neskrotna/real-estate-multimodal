from __future__ import annotations

import re
from pathlib import Path
from typing import List

# Try to enable HEIC/HEIF decoding for Pillow.
# Install once in your venv:
#   pip install pillow-heif
try:
    import pillow_heif  # type: ignore
    pillow_heif.register_heif_opener()
    HEIC_ENABLED = True
except Exception:
    HEIC_ENABLED = False

# Supported image extensions (lowercase)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".jfif", ".heic", ".heif"}


def _natural_key(p: Path):
    """
    Natural sorting key: ensures img_2 < img_10.
    Works well for names like 001_img_01.HEIC, etc.
    """
    parts = re.split(r"(\d+)", p.stem.lower())
    key = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part)
    # include extension in sort stability
    key.append(p.suffix.lower())
    return key


def list_images(folder: str | Path) -> List[Path]:
    """
    Returns a sorted list of image files in the folder.
    - Filters by IMG_EXTS
    - Ignores hidden files
    - Returns [] if folder doesn't exist
    """
    folder = Path(folder)
    if not folder.exists() or not folder.is_dir():
        return []

    paths = [
        p for p in folder.iterdir()
        if p.is_file()
        and not p.name.startswith(".")
        and p.suffix.lower() in IMG_EXTS
    ]

    # Natural sort is safer than plain lexicographic sorting
    return sorted(paths, key=_natural_key)