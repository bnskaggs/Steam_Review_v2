"""Markdown reporting for weekly summaries."""
from __future__ import annotations

import logging
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import duckdb
import pandas as pd

LOGGER = logging.getLogger(__name__)


def _is_duckdb(url: str) -> bool:
    return url.startswith("duckdb://")


def _duckdb_path(url: str) -> Path:
    return Path(url.replace("duckdb://", "", 1))


def _connect(url: str) -> duckdb.DuckDBPyConnection:
    if not _is_duckdb(url):
        raise ValueError("Report currently supports DuckDB only")
    path = _duckdb_path(url)
    if not path.exists():
        raise FileNotFoundError(f"DuckDB database missing: {path}")
    return duckdb.connect(str(path))


def _range_filter() -> str:
    return "WHERE r.ts >= ? AND r.ts < ?"


def _topic_stats(conn: duckdb.DuckDBPyConnection, since: str, until: str) -> pd.DataFrame:
    query = f"""
        SELECT
            t.topic,
            COUNT(DISTINCT CASE WHEN t.sentiment > 0 THEN t.review_id END) AS pos_reviews,
            COUNT(DISTINCT CASE WHEN t.sentiment < 0 THEN t.review_id END) AS neg_reviews,
            COUNT(DISTINCT t.review_id) AS total_reviews,
            AVG(t.confidence) AS avg_conf
        FROM review_topics t
        JOIN reviews r ON r.review_id = t.review_id
        {_range_filter()}
        GROUP BY 1
        ORDER BY total_reviews DESC
    """
    return conn.execute(query, [since, until]).df()


def _daily_counts(conn: duckdb.DuckDBPyConnection, since: str, until: str) -> pd.DataFrame:
    query = f"""
        SELECT
            DATE_TRUNC('day', r.ts) AS day,
            COUNT(*) AS review_count
        FROM reviews r
        {_range_filter()}
        GROUP BY 1
        ORDER BY 1
    """
    return conn.execute(query, [since, until]).df()


def _top_topics(df: pd.DataFrame, sentiment: str, limit: int = 3) -> List[Tuple[str, int]]:
    if df.empty:
        return []
    if sentiment == "positive":
        column = "pos_reviews"
    else:
        column = "neg_reviews"
    filtered = df[df[column] > 0]
    if filtered.empty:
        return []
    sorted_df = filtered.sort_values(by=[column, "total_reviews"], ascending=[False, False])
    top_rows = []
    for _, row in sorted_df.head(limit).iterrows():
        top_rows.append((str(row["topic"]), int(row[column])))
    return top_rows


def _topic_assignments(conn: duckdb.DuckDBPyConnection, since: str, until: str) -> pd.DataFrame:
    query = f"""
        SELECT
            t.review_id,
            t.topic,
            t.sentiment,
            t.confidence,
            COALESCE(r.helpful, 0) AS helpful,
            r.clean_text
        FROM review_topics t
        JOIN reviews r ON r.review_id = t.review_id
        {_range_filter()}
        AND t.sentiment != 0
    """
    return conn.execute(query, [since, until]).df()


def _format_snippet(text: str, limit: int = 180) -> str:
    snippet = re.sub(r"\s+", " ", str(text).strip())
    if not snippet:
        return ""
    if len(snippet) > limit:
        snippet = snippet[: limit - 1].rstrip() + "…"
    return snippet


def _allocate_quotes(assignments: pd.DataFrame) -> Dict[int, Dict[str, List[Tuple[object, str]]]]:
    quote_index: Dict[int, Dict[str, List[Tuple[object, str]]]] = {1: defaultdict(list), -1: defaultdict(list)}
    if assignments.empty:
        return quote_index

    assignments = assignments.dropna(subset=["clean_text"])
    if assignments.empty:
        return quote_index

    assignments["helpful"] = assignments["helpful"].fillna(0)

    for sentiment in (1, -1):
        subset = assignments[assignments["sentiment"] == sentiment]
        if subset.empty:
            continue
        subset = subset.sort_values(
            by=["review_id", "confidence", "helpful"], ascending=[True, False, False]
        )
        subset = subset.drop_duplicates(subset=["review_id"], keep="first")
        subset = subset.sort_values(by=["confidence", "helpful"], ascending=[False, False])

        for _, row in subset.iterrows():
            snippet = _format_snippet(row["clean_text"])
            if not snippet:
                continue
            quote_index[sentiment][row["topic"]].append((row["review_id"], snippet))

    return quote_index


def render(
    db_url: str,
    out_md: str,
    since: str,
    until: str,
    *,
    lang_counts: Dict[str, int] | None = None,
    actual_labeled: float | None = None,
    lang_kept: float | None = None,
) -> Dict[str, object]:
    """Render a Markdown report for the review window."""
    conn = _connect(db_url)
    try:
        daily = _daily_counts(conn, since, until)
        topics = _topic_stats(conn, since, until)
        assignments = _topic_assignments(conn, since, until)
        quote_index = _allocate_quotes(assignments)

        total_reviews = int(daily["review_count"].sum()) if not daily.empty else 0
        pos_topics = _top_topics(topics, "positive")
        neg_topics = _top_topics(topics, "negative")

        lines: List[str] = ["# Weekly Steam Review Report", ""]
        lines.append(f"Window: {since} → {until}")
        lines.append("")
        lines.append(f"Total reviews: {total_reviews}")
        lines.append("")

        if not daily.empty:
            lines.append("## Daily Volume")
            for _, row in daily.iterrows():
                day = row["day"].strftime("%Y-%m-%d") if isinstance(row["day"], datetime) else str(row["day"]).split()[0]
                lines.append(f"- {day}: {int(row['review_count'])}")
            lines.append("")

        def section(title: str, items: List[Tuple[str, int]], sentiment_value: int):
            if not items:
                lines.append(f"## {title}")
                lines.append("No topics matched.")
                lines.append("")
                return
            lines.append(f"## {title}")
            for topic, count in items:
                lines.append(f"- **{topic}** ({int(count)} reviews)")
                quotes = quote_index.get(sentiment_value, {}).get(topic, [])
                seen_reviews = set()
                seen_snippets = set()
                for review_id, snippet in quotes:
                    if review_id in seen_reviews or snippet in seen_snippets:
                        continue
                    lines.append(f"  - \"{snippet}\"")
                    seen_reviews.add(review_id)
                    seen_snippets.add(snippet)
                    if len(seen_reviews) == 3:
                        break
            lines.append("")

        section("Top Positive Topics", pos_topics, 1)
        section("Top Negative Topics", neg_topics, -1)

        lang_counts = lang_counts or {}
        coverage_line = "Classifier coverage: n/a"
        if actual_labeled is not None and lang_kept not in (None, 0):
            included_share = actual_labeled / lang_kept if lang_kept else 0.0
            coverage_line = (
                f"Classifier coverage: {actual_labeled:.1%} of all reviews (~{included_share:.1%} of included languages)"
            )

        lines.append("---")
        lines.append(f"Language mix this window: {lang_counts}")
        lines.append(coverage_line)

        dest = Path(out_md)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("\n".join(lines), encoding="utf-8")
    finally:
        conn.close()

    metrics = {
        "path": str(out_md),
        "top_topics": [topic for topic, _ in (pos_topics + neg_topics)],
        "total_reviews": total_reviews,
        "lang_counts": lang_counts,
        "actual_labeled": actual_labeled,
        "lang_kept": lang_kept,
    }
    LOGGER.info("Report rendered: %s", metrics)
    return metrics


__all__ = ["render"]
