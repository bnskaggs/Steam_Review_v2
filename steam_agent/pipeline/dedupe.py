"""Duplicate removal utilities."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd

LOGGER = logging.getLogger(__name__)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _materialize_key(df: pd.DataFrame, key: str) -> pd.Series:
    parts: List[str] = [part.strip() for part in key.split("|") if part.strip()]
    if not parts:
        raise ValueError("Dedupe key must contain at least one column")

    missing = [part for part in parts if part not in df.columns]
    if missing:
        raise KeyError(f"Columns {missing} missing from dataframe for dedupe key")

    series = df[parts].astype(str)
    return series.apply(lambda row: "|".join(row.values.astype(str)), axis=1)


def dedupe(in_parquet: str, out_parquet: str, key: str = "review_id|clean_text") -> Dict[str, object]:
    """Drop duplicate reviews based on the provided key."""
    src = Path(in_parquet)
    if not src.exists():
        raise FileNotFoundError(in_parquet)

    df = pd.read_parquet(src)
    rows_in = int(df.shape[0])
    if rows_in == 0:
        df["_dedupe_key"] = []
        unique = df
    else:
        df = df.copy()
        df["_dedupe_key"] = _materialize_key(df, key)
        unique = df.drop_duplicates("_dedupe_key").drop(columns=["_dedupe_key"])

    rows_out = int(unique.shape[0])
    dup_rate = float(1 - (rows_out / rows_in)) if rows_in else 0.0

    dest = Path(out_parquet)
    _ensure_parent(dest)
    unique.to_parquet(dest, index=False)

    metrics = {
        "rows_in": rows_in,
        "rows_out": rows_out,
        "dup_rate": round(dup_rate, 4),
        "key": key,
    }
    LOGGER.info("Dedupe complete: %s", metrics)
    return metrics


__all__ = ["dedupe"]
