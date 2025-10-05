"""Materialize processed data into DuckDB and optionally Postgres."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List

import duckdb
import pandas as pd

LOGGER = logging.getLogger(__name__)

_REVIEW_COLUMNS = """
review_id VARCHAR PRIMARY KEY,
app_id INTEGER,
ts TIMESTAMP,
lang TEXT,
clean_text TEXT,
helpful INTEGER,
funny INTEGER,
version_checksum TEXT,
embed_model TEXT,
embedding BLOB
"""

_TOPIC_COLUMNS = """
review_id VARCHAR,
topic TEXT,
sentiment INTEGER,
confidence DOUBLE,
rationale TEXT
"""


def _is_duckdb(url: str) -> bool:
    return url.startswith("duckdb://")


def _duckdb_path(url: str) -> Path:
    path = url.replace("duckdb://", "", 1)
    return Path(path)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _load_parquet(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "embedding" not in df.columns:
        df["embedding"] = None
    return df


def _materialize_duckdb(url: str, reviews: pd.DataFrame, topics: pd.DataFrame) -> None:
    db_path = _duckdb_path(url)
    _ensure_parent(db_path)
    conn = duckdb.connect(str(db_path))
    try:
        reviews_df = reviews.copy()
        topics_df = topics.copy()

        reviews_df["review_id"] = reviews_df["review_id"].astype(str)
        topics_df["review_id"] = topics_df["review_id"].astype(str)

        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS reviews ({_REVIEW_COLUMNS});
            """
        )
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS review_topics ({_TOPIC_COLUMNS});
            """
        )

        conn.register("reviews_stage", reviews_df)
        conn.execute(
            """
            MERGE INTO reviews AS t
            USING reviews_stage AS s
            ON t.review_id = s.review_id
            WHEN MATCHED THEN UPDATE SET
              app_id = s.app_id,
              ts = s.ts,
              lang = s.lang,
              clean_text = s.clean_text,
              helpful = s.helpful,
              funny = s.funny,
              version_checksum = s.version_checksum,
              embed_model = s.embed_model,
              embedding = s.embedding
            WHEN NOT MATCHED THEN
              INSERT (review_id, app_id, ts, lang, clean_text, helpful, funny, version_checksum, embed_model, embedding)
              VALUES (s.review_id, s.app_id, s.ts, s.lang, s.clean_text, s.helpful, s.funny, s.version_checksum, s.embed_model, s.embedding);
            """
        )
        conn.unregister("reviews_stage")

        conn.register("topics_stage", topics_df)
        conn.execute(
            """
            MERGE INTO review_topics AS t
            USING topics_stage AS s
            ON t.review_id = s.review_id AND t.topic = s.topic
            WHEN MATCHED THEN UPDATE SET
              sentiment = s.sentiment,
              confidence = s.confidence,
              rationale = s.rationale
            WHEN NOT MATCHED THEN
              INSERT (review_id, topic, sentiment, confidence, rationale)
              VALUES (s.review_id, s.topic, s.sentiment, s.confidence, s.rationale);
            """
        )
        conn.unregister("topics_stage")
        conn.commit()
    finally:
        conn.close()


def _materialize_postgres(url: str, reviews: pd.DataFrame, topics: pd.DataFrame) -> None:
    try:
        import psycopg
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("psycopg is required for Postgres materialization") from exc

    # Placeholder implementation to keep optional path explicit.
    LOGGER.warning("Postgres materialization not implemented; skipping for url=%s", url)


def load(
    db_url: str,
    reviews_parquet: str,
    topics_parquet: str,
    use_pg: bool = False,
) -> Dict[str, object]:
    """Load processed review data into the configured database."""
    reviews_df = _load_parquet(reviews_parquet)
    topics_df = pd.read_parquet(topics_parquet)

    if _is_duckdb(db_url):
        _materialize_duckdb(db_url, reviews_df, topics_df)
    elif use_pg:
        _materialize_postgres(db_url, reviews_df, topics_df)
    else:
        raise ValueError(f"Unsupported db_url: {db_url}")

    metrics = {"tables": ["reviews", "review_topics"], "db_url": db_url}
    LOGGER.info("Materialization complete: %s", metrics)
    return metrics


__all__ = ["load"]
