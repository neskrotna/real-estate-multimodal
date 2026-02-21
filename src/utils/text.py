from __future__ import annotations

import re

def normalize_text(title: str | None, description: str | None) -> str:
    """
    german-friendly normalization:
    - keep umlauts, capitalization, compounds
    - only collapse whitespace
    """
    title = title or ""
    description = description or ""
    text = (title.strip() + "\n" + description.strip()).strip()
    text = re.sub(r"\s+", " ", text)
    return text