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


def verify_topic_density(
    blank_pct: float,
    lang_kept: float,
    expected_label_rate_en: float = 0.70,
    max_slack: float = 0.20,
) -> bool:
    """Validate topic coverage relative to the detected language mix."""

    expected = lang_kept * expected_label_rate_en
    actual = lang_kept * (1.0 - blank_pct)
    return actual >= (expected - max_slack)


__all__ = ["verify_row_growth", "verify_dup_rate", "verify_topic_density"]
