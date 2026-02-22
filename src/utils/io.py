from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def ensure_parent_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def write_json(path: str | Path, obj: Any) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def read_table(path: str | Path) -> pd.DataFrame:
    """
    Supported formats:
    - .parquet (recommended for processed data)
    - .csv
    - .xlsx / .xls (for raw excel metadata)
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        # Requires pyarrow or fastparquet. pyarrow is preferred.
        return pd.read_parquet(path)

    if suffix == ".csv":
        # UTF-8 by default, but many Excel-export CSVs are UTF-8-SIG.
        try:
            return pd.read_csv(path, encoding="utf-8")
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="utf-8-sig")

    if suffix in [".xlsx", ".xls"]:
        # openpyxl handles .xlsx reliably
        return pd.read_excel(path, engine="openpyxl")

    raise ValueError(f"Unsupported table format: {path}")


def write_parquet(df: pd.DataFrame, path: str | Path) -> None:
    """
    Writes parquet in a way that preserves list columns (like image_paths)
    """
    ensure_parent_dir(path)
    # pyarrow preserves list columns well. If pyarrow isn't installed,
    # pandas will throw an error. That's good (we want it to fail loudly).
    df.to_parquet(path, index=False)