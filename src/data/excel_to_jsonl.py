from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.io import write_jsonl


REQUIRED_COLUMNS = [
    "listing_id",
    "titel",
    "beschreibung_quelle",
    "beschreibung",
    "strasse",
    "postleitzahl",
    "land",
    "stadt",
    "preis_brutto",
    "zimmer",
    "wohnfläche_m2",
    "immobilientyp",
    "zustand",
    "image_folder",
    "image_count",
    "miet_kauf",
    "freifläche",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert listings Excel file to canonical JSONL format."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/descriptions/listings.xlsx"),
        help="Path to the input Excel file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/listings.jsonl"),
        help="Path to the output JSONL file",
    )
    parser.add_argument(
        "--sheet-name",
        default=0,
        help="Excel sheet name or sheet index. Default: first sheet",
    )
    return parser.parse_args()


def normalize_value(value: Any) -> Any:
    if pd.isna(value):
        return None

    if isinstance(value, str):
        value = value.strip()
        return value if value else None

    return value


def normalize_text(value: Any) -> str:
    value = normalize_value(value)
    if value is None:
        return ""
    return str(value).strip()


def normalize_number(value: Any) -> int | float | None:
    value = normalize_value(value)
    if value is None:
        return None

    try:
        number = float(value)
        if number.is_integer():
            return int(number)
        return number
    except (TypeError, ValueError):
        return None


def normalize_country(value: Any) -> str | None:
    value = normalize_text(value).upper()
    if not value:
        return None

    mapping = {
        "AT": "AT",
        "AUT": "AT",
        "AUSTRIA": "AT",
        "ÖSTERREICH": "AT",
        "OESTERREICH": "AT",
        "SK": "SK",
        "SVK": "SK",
        "SLOVAKIA": "SK",
        "SLOWAKEI": "SK",
    }

    return mapping.get(value, value)


def normalize_text_source(value: Any) -> str | None:
    value = normalize_text(value).lower()
    if not value:
        return None

    mapping = {
        "synthetisch": "synthetic",
        "synthetic": "synthetic",
        "original": "original",
        "original_übersetzt": "translated",
        "original_uebersetzt": "translated",
        "übersetzt": "translated",
        "uebersetzt": "translated",
        "translated": "translated",
    }

    return mapping.get(value, value)


def normalize_listing_type(value: Any) -> str | None:
    value = normalize_text(value).lower()
    if not value:
        return None

    mapping = {
        "miete": "rent",
        "kauf": "sale",
        "rent": "rent",
        "sale": "sale",
    }

    return mapping.get(value, value)


def normalize_bool_ja_nein(value: Any) -> bool | None:
    value = normalize_text(value).lower()
    if not value:
        return None

    if value in {"ja", "yes", "true", "1"}:
        return True
    if value in {"nein", "no", "false", "0"}:
        return False

    return None


def validate_required_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required Excel columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def build_record(row: pd.Series) -> dict[str, Any]:
    record = {
        "listing_id": normalize_text(row.get("listing_id")),
        "title": normalize_text(row.get("titel")),
        "text_source": normalize_text_source(row.get("beschreibung_quelle")),
        "description": normalize_text(row.get("beschreibung")),
        "address": normalize_text(row.get("strasse")),
        "postal_code": normalize_text(row.get("postleitzahl")),
        "country": normalize_country(row.get("land")),
        "city": normalize_text(row.get("stadt")),
        "price_eur": normalize_number(row.get("preis_brutto")),
        "rooms": normalize_number(row.get("zimmer")),
        "area_m2": normalize_number(row.get("wohnfläche_m2")),
        "property_type": normalize_text(row.get("immobilientyp")),
        "condition": normalize_text(row.get("zustand")),
        "image_folder": normalize_text(row.get("image_folder")),
        "image_count": normalize_number(row.get("image_count")),
        "listing_type": normalize_listing_type(row.get("miet_kauf")),
        "has_outdoor_space": normalize_bool_ja_nein(row.get("freifläche")),
    }

    return record


def main() -> None:
    args = parse_args()

    df = pd.read_excel(args.input, sheet_name=args.sheet_name)
    validate_required_columns(df)

    records: list[dict[str, Any]] = []
    skipped_missing_id = 0

    for _, row in df.iterrows():
        record = build_record(row)

        if not record["listing_id"]:
            skipped_missing_id += 1
            continue

        records.append(record)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, records)

    print(f"[INFO] Read {len(df)} rows from {args.input}")
    print(f"[INFO] Wrote {len(records)} records to {args.output}")

    if skipped_missing_id > 0:
        print(f"[INFO] Skipped {skipped_missing_id} rows without listing_id")


if __name__ == "__main__":
    main()