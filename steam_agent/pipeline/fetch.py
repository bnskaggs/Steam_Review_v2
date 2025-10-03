"""Steam review fetching utilities."""
from __future__ import annotations

import csv
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
import requests
from requests import Response

LOGGER = logging.getLogger(__name__)

_API_URL = "https://store.steampowered.com/appreviews/{app_id}"
_DEFAULT_PARAMS = {
    "json": 1,
    "num_per_page": 100,
    "language": "all",
    "purchase_type": "all",
    "review_type": "all",
    "filter": "recent",
}


@dataclass
class _Batch:
    cursor: str
    rows: List[Dict[str, object]]


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _request(app_id: int, cursor: str) -> Response:
    params = dict(_DEFAULT_PARAMS)
    params["cursor"] = cursor
    url = _API_URL.format(app_id=app_id)
    LOGGER.debug("Fetching cursor %s", cursor)
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response


def _parse_timestamp(ts: int) -> datetime:
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def _should_continue(batch: Dict[str, object], since_dt: datetime) -> bool:
    reviews = batch.get("reviews", [])
    if not reviews:
        return False
    oldest = min(_parse_timestamp(r["timestamp_created"]) for r in reviews)
    return oldest >= since_dt


def _extract_rows(batch: Dict[str, object], since_dt: datetime) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for raw in batch.get("reviews", []):
        ts = _parse_timestamp(raw["timestamp_created"])
        if ts < since_dt:
            continue
        rows.append(
            {
                "review_id": raw.get("recommendationid"),
                "app_id": raw.get("appid", 0),
                "timestamp_created": ts.isoformat(),
                "language": raw.get("language"),
                "review": raw.get("review", ""),
                "votes_helpful": raw.get("votes_up", 0),
                "votes_funny": raw.get("votes_funny", 0),
                "weighted_vote_score": raw.get("weighted_vote_score", 0.0),
                "comment_count": raw.get("comment_count", 0),
            }
        )
    return rows


def _iterate_batches(app_id: int, since_dt: datetime) -> Iterable[_Batch]:
    cursor = "*"
    while True:
        resp = _request(app_id, cursor)
        payload = resp.json()
        rows = _extract_rows(payload, since_dt)
        yield _Batch(cursor=cursor, rows=rows)
        cursor = payload.get("cursor", cursor)
        if not cursor or cursor == "\u2603":
            break
        if not _should_continue(payload, since_dt):
            break
        time.sleep(0.5)


def fetch_reviews(app_id: int, since: str, out_csv: str) -> Dict[str, object]:
    """Fetch recent Steam reviews and persist them as CSV.

    Parameters
    ----------
    app_id:
        Steam application identifier.
    since:
        Earliest timestamp (ISO format) to retain.
    out_csv:
        Output CSV path.
    """
    since_dt = datetime.fromisoformat(since).astimezone(timezone.utc)
    dest = Path(out_csv)
    _ensure_parent(dest)

    rows: List[Dict[str, object]] = []
    for batch in _iterate_batches(app_id, since_dt):
        rows.extend(batch.rows)
        LOGGER.info("Fetched %s rows for cursor %s", len(batch.rows), batch.cursor)
        if len(batch.rows) < _DEFAULT_PARAMS["num_per_page"]:
            LOGGER.debug("Short batch encountered; likely done")
            break

    if rows:
        df = pd.DataFrame(rows).sort_values("timestamp_created")
    else:
        df = pd.DataFrame(
            columns=[
                "review_id",
                "app_id",
                "timestamp_created",
                "language",
                "review",
                "votes_helpful",
                "votes_funny",
                "weighted_vote_score",
                "comment_count",
            ]
        )
    df.to_csv(dest, index=False, quoting=csv.QUOTE_MINIMAL)

    metrics = {
        "rows": int(df.shape[0]),
        "file": str(dest),
        "since": since,
        "app_id": app_id,
    }
    LOGGER.info("Fetch complete: %s", metrics)
    return metrics


__all__ = ["fetch_reviews"]
