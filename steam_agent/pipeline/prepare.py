"""Data cleaning and preparation stage."""
from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict
import pandas as pd

LOGGER = logging.getLogger(__name__)

_WHITESPACE_RE = re.compile(r"\s+")
_MARKUP_RE = re.compile(r"<[^>]+>")


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _clean_text(value: str) -> str:
    if not isinstance(value, str):
        return ""
    lowered = value.lower()
    without_markup = _MARKUP_RE.sub(" ", lowered)
    normalized = _WHITESPACE_RE.sub(" ", without_markup)
    return normalized.strip()


def _parse_timestamp(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _checksum(clean_text: str, ts: datetime, app_id: int) -> str:
    sha = hashlib.sha1()
    sha.update(clean_text.encode("utf-8"))
    sha.update(str(app_id).encode("utf-8"))
    sha.update(ts.isoformat().encode("utf-8"))
    return sha.hexdigest()


def prepare(in_csv: str, out_parquet: str, lang: str = "english") -> Dict[str, object]:
    """Prepare raw review CSV into a canonical Parquet dataset."""
    src = Path(in_csv)
    if not src.exists():
        raise FileNotFoundError(in_csv)

    LOGGER.info("Loading raw CSV %s", src)
    df = pd.read_csv(src)
    rows_in = int(df.shape[0])

    kept = 0
    if rows_in == 0:
        cleaned = pd.DataFrame(
            columns=
            [
                "review_id",
                "app_id",
                "ts",
                "lang",
                "clean_text",
                "helpful",
                "funny",
                "version_checksum",
                "embed_model",
            ]
        )
    else:
        df["language"] = df["language"].fillna("").str.lower()
        df["lang_match"] = df["language"] == lang.lower()
        kept = int(df["lang_match"].sum())

        df["ts"] = df["timestamp_created"].apply(lambda x: _parse_timestamp(str(x)))
        df["clean_text"] = df["review"].apply(_clean_text)
        df["helpful"] = df["votes_helpful"].fillna(0).astype(int)
        df["funny"] = df["votes_funny"].fillna(0).astype(int)
        df["app_id"] = df["app_id"].fillna(0).astype(int)

        df["version_checksum"] = df.apply(
            lambda row: _checksum(row["clean_text"], row["ts"], row["app_id"]),
            axis=1,
        )
        cleaned = df.loc[df["lang_match"], [
            "review_id",
            "app_id",
            "ts",
            "language",
            "clean_text",
            "helpful",
            "funny",
            "version_checksum",
        ]].rename(columns={"language": "lang"})
        cleaned["embed_model"] = "none"

    pct_kept = float(kept / rows_in) if rows_in else 0.0

    dest = Path(out_parquet)
    _ensure_parent(dest)
    cleaned.to_parquet(dest, index=False)

    metrics = {
        "rows_in": rows_in,
        "rows_out": int(cleaned.shape[0]),
        "pct_lang_kept": round(pct_kept, 4),
    }
    LOGGER.info("Prepare complete: %s", metrics)
    return metrics


__all__ = ["prepare"]
