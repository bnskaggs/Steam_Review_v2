"""Embedding stub stage."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import pandas as pd

LOGGER = logging.getLogger(__name__)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def embed(in_parquet: str, out_parquet: str, model: str = "none", force: bool = False) -> Dict[str, object]:
    """Passthrough embedding stage.

    Parameters
    ----------
    model: str
        Name of the embedding model. When "none", no embeddings are generated.
    force: bool
        Placeholder to allow callers to trigger re-embedding in the future.
    """
    src = Path(in_parquet)
    if not src.exists():
        raise FileNotFoundError(in_parquet)

    df = pd.read_parquet(src).copy()
    rows_in = int(df.shape[0])

    if model == "none":
        coverage = 1.0 if rows_in else 0.0
        df["embed_model"] = "none"
        if "embedding" not in df.columns:
            df["embedding"] = None
    else:
        coverage = 0.0
        df["embed_model"] = model
        df["embedding"] = None

    dest = Path(out_parquet)
    _ensure_parent(dest)
    df.to_parquet(dest, index=False)

    metrics = {
        "rows_in": rows_in,
        "rows_out": int(df.shape[0]),
        "coverage": coverage,
        "model": model,
        "force": force,
    }
    LOGGER.info("Embed stage complete: %s", metrics)
    return metrics


__all__ = ["embed"]
