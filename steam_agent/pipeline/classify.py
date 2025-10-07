"""Rule-based review classification."""
from __future__ import annotations

import logging
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml

LOGGER = logging.getLogger(__name__)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _load_taxonomy(path: Path) -> Dict[str, List[str]]:
    """Load topic → keyword mapping from YAML."""
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    topics = payload.get("topics", {})
    return {str(topic): [str(kw).lower() for kw in kws] for topic, kws in topics.items()}


def _build_topic_patterns(taxonomy: Dict[str, List[str]]) -> Dict[str, re.Pattern]:
    """Compile flexible regex patterns for each topic."""
    patterns = {}
    for topic, keywords in taxonomy.items():
        if not keywords:
            continue
        # Build pattern with word boundaries
        escaped = [re.escape(kw) for kw in keywords]
        pattern_str = r"(?:^|[^a-z0-9_])(" + "|".join(escaped) + r")(?:[^a-z0-9_]|$)"
        patterns[topic] = re.compile(pattern_str, re.I)
    return patterns


def _classify_review(text: str, patterns: Dict[str, re.Pattern]) -> List[Dict[str, object]]:
    """Classify a single review against all topics."""
    topic_hits: Dict[str, int] = {}

    for topic, pattern in patterns.items():
        matches = pattern.findall(text)
        if matches:
            topic_hits[topic] = len(matches)

    if not topic_hits:
        return []

    # Calculate confidence proportional to hits
    total_hits = sum(topic_hits.values())
    results = []
    for topic, hits in topic_hits.items():
        confidence = min(1.0, hits / 5.0)  # Cap at 1.0, scale by 5 hits = full confidence
        results.append({
            "topic": topic,
            "confidence": round(confidence, 4),
            "hits": hits,
        })

    return results


def _density_report(df: pd.DataFrame, total_reviews: int) -> str:
    """Generate topic coverage report."""
    if df.empty or total_reviews == 0:
        return "No reviews classified."

    topic_counts = Counter(df["topic"].tolist())

    lines = ["┌────────────────────────────┬────────────┐"]
    lines.append("│ Topic                      │ Coverage % │")
    lines.append("├────────────────────────────┼────────────┤")

    for topic in sorted(topic_counts.keys()):
        count = topic_counts[topic]
        pct = 100.0 * count / total_reviews
        topic_padded = topic[:26].ljust(26)
        pct_str = f"{pct:.1f}%".rjust(10)
        lines.append(f"│ {topic_padded} │ {pct_str} │")

    lines.append("└────────────────────────────┴────────────┘")

    unique_reviews = df["review_id"].nunique()
    overall_pct = 100.0 * unique_reviews / total_reviews
    lines.append(f"\nTotal: {unique_reviews}/{total_reviews} reviews classified ({overall_pct:.1f}%)")

    return "\n".join(lines)


def classify(
    in_parquet: str,
    out_parquet: str,
    taxonomy: str,
) -> Dict[str, object]:
    """Assign topics to reviews using keyword matching."""
    src = Path(in_parquet)
    if not src.exists():
        raise FileNotFoundError(in_parquet)

    tax_path = Path(taxonomy)
    if not tax_path.exists():
        raise FileNotFoundError(taxonomy)

    taxonomy_map = _load_taxonomy(tax_path)
    patterns = _build_topic_patterns(taxonomy_map)

    df = pd.read_parquet(src)
    rows_in = int(df.shape[0])

    records: List[Dict[str, object]] = []

    for idx, row in df.iterrows():
        review_id = row.get("review_id")

        # Use clean_text if available, fall back to review
        if "clean_text" in row and pd.notna(row["clean_text"]):
            text = str(row["clean_text"])
        elif "review" in row and pd.notna(row["review"]):
            text = str(row["review"])
        else:
            text = ""

        if not text.strip():
            continue

        classifications = _classify_review(text, patterns)

        for cls in classifications:
            records.append({
                "review_id": review_id,
                "topic": cls["topic"],
                "confidence": cls["confidence"],
                "hits": cls["hits"],
            })

    if records:
        labeled_df = pd.DataFrame(records)
    else:
        labeled_df = pd.DataFrame(columns=["review_id", "topic", "confidence", "hits"])

    dest = Path(out_parquet)
    _ensure_parent(dest)
    labeled_df.to_parquet(dest, index=False)

    # Print density report
    density_table = _density_report(labeled_df, rows_in)
    print("\n" + density_table + "\n")
    LOGGER.info("\n%s", density_table)

    labeled_reviews = set(labeled_df["review_id"].dropna().tolist()) if not labeled_df.empty else set()
    rows_labeled = len(labeled_reviews)
    pct_labeled = (rows_labeled / rows_in) if rows_in else 0.0
    avg_conf = float(labeled_df["confidence"].mean()) if not labeled_df.empty else 0.0

    metrics = {
        "rows_in": rows_in,
        "rows_labeled": rows_labeled,
        "pct_labeled": round(pct_labeled, 4),
        "avg_conf": round(avg_conf, 4),
    }
    LOGGER.info("Classification complete: %s", metrics)
    return metrics


__all__ = ["classify"]
