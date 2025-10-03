"""Markdown reporting for weekly summaries."""
from __future__ import annotations

import logging
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
            SUM(CASE WHEN t.sentiment > 0 THEN 1 ELSE 0 END) AS pos_reviews,
            SUM(CASE WHEN t.sentiment < 0 THEN 1 ELSE 0 END) AS neg_reviews,
            COUNT(*) AS total_reviews,
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
    sorted_df = filtered.sort_values(by=[column, "total_reviews"], ascending=[False, False])
    return list(zip(sorted_df["topic"].head(limit), sorted_df[column].head(limit)))


def _topic_quotes(
    conn: duckdb.DuckDBPyConnection,
    topic: str,
    since: str,
    until: str,
    limit: int = 3,
) -> List[str]:
    query = f"""
        SELECT r.clean_text
        FROM review_topics t
        JOIN reviews r ON r.review_id = t.review_id
        {_range_filter()} AND t.topic = ?
        ORDER BY r.helpful DESC
        LIMIT {limit}
    """
    rows = conn.execute(query, [since, until, topic]).fetchall()
    return [row[0] for row in rows if row and row[0]]


def render(db_url: str, out_md: str, since: str, until: str) -> Dict[str, object]:
    """Render a Markdown report for the review window."""
    conn = _connect(db_url)
    try:
        daily = _daily_counts(conn, since, until)
        topics = _topic_stats(conn, since, until)

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

        def section(title: str, items: List[Tuple[str, int]]):
            if not items:
                lines.append(f"## {title}")
                lines.append("No topics matched.")
                lines.append("")
                return
            lines.append(f"## {title}")
            for topic, count in items:
                lines.append(f"- **{topic}** ({count} reviews)")
                quotes = _topic_quotes(conn, topic, since, until)
                for quote in quotes:
                    snippet = quote[:200].replace("\n", " ")
                    lines.append(f"  - \"{snippet}\"")
            lines.append("")

        section("Top Positive Topics", pos_topics)
        section("Top Negative Topics", neg_topics)

        dest = Path(out_md)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("\n".join(lines), encoding="utf-8")
    finally:
        conn.close()

    metrics = {
        "path": str(out_md),
        "top_topics": [topic for topic, _ in (pos_topics + neg_topics)],
        "total_reviews": total_reviews,
    }
    LOGGER.info("Report rendered: %s", metrics)
    return metrics


__all__ = ["render"]
