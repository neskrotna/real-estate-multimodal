from __future__ import annotations


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def build_listing_text(row: dict) -> str:
    parts = []

    title = str(row.get("title", "")).strip()
    description = str(row.get("description", "")).strip()
    city = str(row.get("city", "")).strip()
    property_type = str(row.get("property_type", "")).strip()
    condition = str(row.get("condition", "")).strip()
    listing_type = str(row.get("listing_type", "")).strip()

    if title:
        parts.append(title)

    if description:
        parts.append(description)

    structured = []
    if city:
        structured.append(f"Stadt: {city}")
    if property_type:
        structured.append(f"Typ: {property_type}")
    if condition:
        structured.append(f"Zustand: {condition}")
    if listing_type:
        structured.append(f"Angebot: {listing_type}")

    if structured:
        parts.append(" | ".join(structured))

    return normalize_whitespace(" ".join(parts))