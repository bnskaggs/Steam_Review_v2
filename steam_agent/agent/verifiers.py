"""Pure verification checks for pipeline metrics."""
from __future__ import annotations

from typing import Iterable, Sequence, Tuple


def verify_row_growth(
    prev_count: int | None,
    new_count: int,
    min_growth_pct: float = 0.02,
) -> Tuple[bool, str]:
    if prev_count is None or prev_count == 0:
        return True, "no baseline"
    growth = (new_count - prev_count) / prev_count
    if growth < min_growth_pct:
        return False, f"row growth {growth:.4f} below threshold {min_growth_pct:.4f}"
    return True, f"row growth {growth:.4f}"


def verify_lang_mix(max_non_english_pct: float = 0.10, pct_lang_kept: float | None = None) -> Tuple[bool, str]:
    if pct_lang_kept is None:
        return True, "no language metric"
    non_english = 1 - pct_lang_kept
    if non_english > max_non_english_pct:
        return False, f"non-english ratio {non_english:.4f} exceeds {max_non_english_pct:.4f}"
    return True, f"non-english ratio {non_english:.4f}"


def verify_dup_rate(dup_rate: float, max_dup_pct: float = 0.05) -> Tuple[bool, str]:
    if dup_rate > max_dup_pct:
        return False, f"dup_rate {dup_rate:.4f} exceeds {max_dup_pct:.4f}"
    return True, f"dup_rate {dup_rate:.4f}"


def verify_embed_coverage(coverage: float, min_pct: float = 0.99) -> Tuple[bool, str]:
    if coverage < min_pct:
        return False, f"embed coverage {coverage:.4f} below {min_pct:.4f}"
    return True, f"embed coverage {coverage:.4f}"


def verify_topic_consistency(
    min_conf: float,
    avg_conf: float,
    blank_pct: float,
    max_blank_pct: float = 0.10,
) -> Tuple[bool, str]:
    if avg_conf < min_conf:
        return False, f"avg_conf {avg_conf:.4f} below min_conf {min_conf:.4f}"
    if blank_pct > max_blank_pct:
        return False, f"blank_pct {blank_pct:.4f} exceeds {max_blank_pct:.4f}"
    return True, f"avg_conf {avg_conf:.4f}, blank_pct {blank_pct:.4f}"


def verify_views_materialized(tables: Sequence[str], required: Sequence[str] = ("reviews", "review_topics")) -> Tuple[bool, str]:
    missing = [table for table in required if table not in tables]
    if missing:
        return False, f"missing tables: {', '.join(missing)}"
    return True, "all tables materialized"


__all__ = [
    "verify_row_growth",
    "verify_lang_mix",
    "verify_dup_rate",
    "verify_embed_coverage",
    "verify_topic_consistency",
    "verify_views_materialized",
]
