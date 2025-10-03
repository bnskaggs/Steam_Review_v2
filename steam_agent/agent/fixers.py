"""Targeted remediation helpers."""
from __future__ import annotations


def tighten_dedupe_key(prev_key: str) -> str:
    parts = [segment.strip() for segment in prev_key.split("|") if segment.strip()]
    if "version_checksum" not in parts:
        parts.append("version_checksum")
    elif "ts" not in parts:
        parts.append("ts")
    else:
        parts.append("clean_text")
    return "|".join(dict.fromkeys(parts))


def lower_topic_threshold(current: float, floor: float = 0.45, step: float = 0.05) -> float:
    new_threshold = max(floor, current - step)
    return round(new_threshold, 4)


def force_reembed() -> dict:
    return {"force": True}


__all__ = ["tighten_dedupe_key", "lower_topic_threshold", "force_reembed"]
