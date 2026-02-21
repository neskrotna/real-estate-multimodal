from __future__ import annotations

from pathlib import Path
from typing import List

IMG_EXTS = {".jpg", ".jpeg", ".png", ".HEIC"}

def list_images(folder: str | Path) -> List[Path]:
    folder = Path(folder)
    if not folder.exists():
        return []
    paths = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    return sorted(paths)