"""Remediation helpers for pipeline retries."""
from __future__ import annotations


def tighten_dedupe_key(prev_key: str = "review_id|clean_text") -> str:
    """Broaden the dedupe key slightly when duplicate rate is too high."""
    pieces = [part.strip() for part in prev_key.split("|") if part.strip()]
    if "version_checksum" not in pieces:
        pieces.append("version_checksum")
    elif "ts" not in pieces:
        pieces.append("ts")
    else:
        pieces.append("review_id")
    return "|".join(dict.fromkeys(pieces))


def force_reembed() -> dict:
    """Return kwargs that force the embed stage to rebuild vectors."""
    return {"force": True}


__all__ = ["tighten_dedupe_key", "force_reembed"]
