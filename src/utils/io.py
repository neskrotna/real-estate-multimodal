from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd


def ensure_parent_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: Any) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def write_jsonl(path: str | Path, records: Iterable[Dict[str, Any]]) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_table(path: str | Path) -> pd.DataFrame:
    """
    Supported formats:
    - .parquet
    - .csv
    - .xlsx / .xls
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        return pd.read_parquet(path)

    if suffix == ".csv":
        try:
            return pd.read_csv(path, encoding="utf-8")
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="utf-8-sig")

    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(path, engine="openpyxl")

    raise ValueError(f"Unsupported table format: {path}")


def write_parquet(df: pd.DataFrame, path: str | Path) -> None:
    ensure_parent_dir(path)
    df.to_parquet(path, index=False)