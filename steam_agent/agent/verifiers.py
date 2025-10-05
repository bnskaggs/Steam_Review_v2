"""Boolean verifiers for pipeline metrics."""
from __future__ import annotations


def verify_row_growth(prev: int | None, new: int, min_growth: float = 0.02) -> bool:
    """Return True when the new fetch grows by the required percentage."""
    if prev is None:
        return new > 0
    if prev == 0:
        return new >= 0
    growth = (new - prev) / prev
    return growth >= min_growth


def verify_dup_rate(dup_rate: float, max_dup: float = 0.05) -> bool:
    """Return True when the duplicate ratio is within tolerance."""
    return dup_rate <= max_dup


def verify_topic_density(blank_pct: float, max_blank: float = 0.50) -> bool:
    """Return True when at least half the reviews receive a topic label."""
    return blank_pct <= max_blank


__all__ = ["verify_row_growth", "verify_dup_rate", "verify_topic_density"]
