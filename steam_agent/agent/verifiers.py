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
    lang_counts: dict[str, int],
    supported_langs: set[str] = {"en"},
    target_supported_rate: float = 0.70,
    max_abs_slack: float = 0.20,
    min_rows_for_strict: int = 500,
    total_rows: int | None = None,
) -> tuple[bool, dict[str, float | bool]]:
    """Validate topic coverage relative to the detected language mix."""

    total = sum(lang_counts.values()) or 1
    supported = sum(lang_counts.get(lang, 0) for lang in supported_langs)
    supported_share = supported / total
    actual_labeled = 1.0 - blank_pct
    expected_overall = supported_share * target_supported_rate
    strict = (total_rows or total) >= min_rows_for_strict
    tolerance = max_abs_slack if strict else max_abs_slack + 0.10
    passed = actual_labeled >= (expected_overall - tolerance)
    return passed, {
        "supported_share": supported_share,
        "expected_overall": expected_overall,
        "actual_labeled": actual_labeled,
        "strict": strict,
    }


__all__ = ["verify_row_growth", "verify_dup_rate", "verify_topic_density"]
