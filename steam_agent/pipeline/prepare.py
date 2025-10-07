"""Data cleaning and preparation stage."""
from __future__ import annotations

import hashlib
import logging
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

LOGGER = logging.getLogger(__name__)

_WHITESPACE_RE = re.compile(r"\s+")
_MARKUP_RE = re.compile(r"<[^>]+>")

_EN_STOPWORDS = {"the", "and", "is", "this", "that", "with", "for", "not", "you"}
_ES_STOPWORDS = {"el", "la", "los", "las", "que", "con", "para", "sin", "es"}
_ACCENTED_CHARS = "áéíóúñü"

_LANG_HINTS: Dict[str, str] = {
    "english": "en",
    "spanish": "es",
    "latam": "es",
    "latamspanish": "es",
    "german": "de",
    "french": "fr",
    "italian": "it",
    "portuguese": "pt",
    "brazilian": "pt",
    "schinese": "zh",
    "tchinese": "zh",
    "japanese": "ja",
    "koreana": "ko",
}


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


def _normalize_lang_code(value: str | float | None) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = str(value).strip().lower()
    if not text:
        return ""
    if text in _LANG_HINTS:
        return _LANG_HINTS[text]
    if "-" in text:
        text = text.split("-", 1)[0]
    if len(text) == 2:
        return text
    return text[:2]


def infer_language(text: str, provided_lang: str | None = None) -> str:
    """Infer a short language code using lightweight heuristics."""
    normalized_hint = _normalize_lang_code(provided_lang)
    if normalized_hint:
        return normalized_hint

    if not isinstance(text, str):
        return "unknown"

    lowered = text.lower()
    words = re.findall(r"[a-záéíóúñü]+", lowered)
    if not words:
        return "unknown"

    en_hits = sum(1 for word in words if word in _EN_STOPWORDS)
    es_hits = sum(1 for word in words if word in _ES_STOPWORDS)

    accent_chars = sum(1 for ch in lowered if ch in _ACCENTED_CHARS)
    alpha_chars = sum(1 for ch in lowered if ch.isalpha()) or 1
    accent_ratio = accent_chars / alpha_chars

    if es_hits > en_hits or accent_ratio > 0.05:
        return "es"
    if en_hits >= es_hits:
        return "en"
    return "unknown"


def prepare(
    in_csv: str,
    out_parquet: str,
    langs: List[str] | None = ["en"],
) -> Dict[str, object]:
    """Prepare raw review CSV into a canonical Parquet dataset."""
    src = Path(in_csv)
    if not src.exists():
        raise FileNotFoundError(in_csv)

    LOGGER.info("Loading raw CSV %s", src)
    df = pd.read_csv(src)
    rows_in = int(df.shape[0])

    lang_counts: Dict[str, int] = {}
    lang_counts_raw: Counter[str] = Counter()
    lang_counts_kept: Counter[str] = Counter()
    if rows_in == 0:
        cleaned = pd.DataFrame(
            columns=[
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
        rows_out = 0
    else:
        review_texts = df.get("review", pd.Series([""] * rows_in))
        lang_hints = df.get("language", pd.Series([None] * rows_in))
        detected_langs: List[str] = []
        for text, hint in zip(review_texts, lang_hints):
            normalized_text = "" if (not isinstance(text, str) and pd.isna(text)) else str(text)
            detected_langs.append(infer_language(normalized_text, hint))
        df["lang"] = detected_langs
        raw_counter = Counter(detected_langs)
        lang_counts_raw = Counter(raw_counter)
        lang_counts = {key: int(value) for key, value in raw_counter.items()}

        whitelist: Optional[set[str]]
        if langs is None:
            whitelist = None
        else:
            whitelist = {code.lower() for code in langs if code}
        if whitelist is not None:
            lang_mask = df["lang"].isin(whitelist)
        else:
            lang_mask = pd.Series([True] * rows_in)

        filtered = df.loc[lang_mask].copy()
        rows_out = int(filtered.shape[0])

        if rows_out == 0:
            cleaned = pd.DataFrame(
                columns=[
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
            lang_counts_kept = Counter()
        else:
            filtered["ts"] = filtered["timestamp_created"].apply(lambda x: _parse_timestamp(str(x)))
            filtered["clean_text"] = filtered["review"].apply(_clean_text)
            filtered["helpful"] = filtered["votes_helpful"].fillna(0).astype(int)
            filtered["funny"] = filtered["votes_funny"].fillna(0).astype(int)
            filtered["app_id"] = filtered["app_id"].fillna(0).astype(int)

            filtered["version_checksum"] = filtered.apply(
                lambda row: _checksum(row["clean_text"], row["ts"], row["app_id"]),
                axis=1,
            )
            cleaned = filtered[[
                "review_id",
                "app_id",
                "ts",
                "lang",
                "clean_text",
                "helpful",
                "funny",
                "version_checksum",
            ]].copy()
            cleaned.loc[:, "embed_model"] = "none"
            lang_counts_kept = Counter(filtered["lang"])

    pct_kept = float(rows_out / rows_in) if rows_in else 0.0

    dest = Path(out_parquet)
    _ensure_parent(dest)
    cleaned.to_parquet(dest, index=False)

    metrics = {
        "rows_in": rows_in,
        "rows_out": rows_out,
        "rows_in_raw": rows_in,
        "rows_in_kept": rows_out,
        "pct_lang_kept": round(pct_kept, 4),
        "lang_counts": lang_counts,
        "lang_counts_raw": lang_counts_raw,
        "lang_counts_kept": lang_counts_kept,
    }
    LOGGER.info("Prepare complete: %s", metrics)
    return metrics


__all__ = ["prepare"]
