"""Rule-based review classification."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml

LOGGER = logging.getLogger(__name__)

_POSITIVE = {
    "good",
    "great",
    "love",
    "fun",
    "awesome",
    "amazing",
    "smooth",
    "fast",
    "responsive",
    "enjoy",
    "solid",
}
_NEGATIVE = {
    "bad",
    "terrible",
    "hate",
    "buggy",
    "lag",
    "slow",
    "broken",
    "crash",
    "boring",
    "grindy",
    "paywall",
    "expensive",
    "stutter",
}


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _load_taxonomy(path: Path) -> Dict[str, List[str]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    topics = payload.get("topics", {})
    return {str(topic): [str(kw).lower() for kw in kws] for topic, kws in topics.items()}


def _score_topic(text: str, keywords: List[str]) -> Dict[str, object] | None:
    if not text:
        return None
    lowered = text.lower()
    hits: List[str] = []
    for kw in keywords:
        if kw and kw in lowered:
            hits.append(kw)
    if not hits:
        return None

    tokens = lowered.split()
    windows = max(1, len(tokens) // 20)
    score = min(1.0, len(hits) / windows)
    positive = any(token in _POSITIVE for token in tokens)
    negative = any(token in _NEGATIVE for token in tokens)
    if positive and not negative:
        sentiment = 1
    elif negative and not positive:
        sentiment = -1
    elif positive and negative:
        sentiment = 0
    else:
        sentiment = 0

    rationale = f"hits: {', '.join(sorted(set(hits)))}"
    return {"confidence": score, "sentiment": sentiment, "rationale": rationale}


def classify(
    in_parquet: str,
    out_parquet: str,
    taxonomy: str,
    min_conf: float = 0.55,
) -> Dict[str, object]:
    """Assign topics to reviews using keyword matching."""
    src = Path(in_parquet)
    if not src.exists():
        raise FileNotFoundError(in_parquet)

    tax_path = Path(taxonomy)
    if not tax_path.exists():
        raise FileNotFoundError(taxonomy)

    taxonomy_map = _load_taxonomy(tax_path)
    df = pd.read_parquet(src)
    rows_in = int(df.shape[0])

    labels: List[Dict[str, object]] = []
    reviews_with_labels = set()

    for _, row in df.iterrows():
        text = row.get("clean_text", "")
        for topic, keywords in taxonomy_map.items():
            score = _score_topic(text, keywords)
            if not score:
                continue
            if score["confidence"] < min_conf:
                continue
            labels.append(
                {
                    "review_id": row.get("review_id"),
                    "topic": topic,
                    "sentiment": score["sentiment"],
                    "confidence": round(float(score["confidence"]), 4),
                    "rationale": score["rationale"],
                }
            )
            reviews_with_labels.add(row.get("review_id"))

    if labels:
        labeled_df = pd.DataFrame(labels)
    else:
        labeled_df = pd.DataFrame(columns=["review_id", "topic", "sentiment", "confidence", "rationale"])

    dest = Path(out_parquet)
    _ensure_parent(dest)
    labeled_df.to_parquet(dest, index=False)

    rows_labeled = int(labeled_df.shape[0])
    blank_pct = float(1 - (len(reviews_with_labels) / rows_in)) if rows_in else 0.0
    avg_conf = float(labeled_df["confidence"].mean()) if rows_labeled else 0.0

    metrics = {
        "rows_in": rows_in,
        "rows_labeled": rows_labeled,
        "blank_pct": round(blank_pct, 4),
        "avg_conf": round(avg_conf, 4),
        "min_conf": min_conf,
    }
    LOGGER.info("Classification complete: %s", metrics)
    return metrics


__all__ = ["classify"]
