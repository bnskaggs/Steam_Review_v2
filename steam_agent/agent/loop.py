"""Pipeline operator agent loop."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from rich.console import Console
from rich.table import Table

from ..pipeline import classify, dedupe, embed, fetch, materialize, prepare, report
from . import fixers, verifiers

LOGGER = logging.getLogger(__name__)
console = Console()


@dataclass
class RunConfig:
    app_id: int
    since: str
    until: str
    raw: str
    clean: str
    unique: str
    embedded: str
    topics: str
    db_url: str
    report: str
    taxonomy: str = str(Path(__file__).resolve().parent / "taxonomy.yaml")
    embed_model: str = "none"
    min_conf: float = 0.55
    use_pg: bool = False
    max_reviews: Optional[int] = None

    def as_dict(self) -> Dict[str, object]:
        return self.__dict__


@dataclass
class StepResult:
    name: str
    metrics: Dict[str, object]
    verifier_status: str = "pending"


def _existing_review_count(db_url: str) -> Optional[int]:
    if not db_url.startswith("duckdb://"):
        return None
    path = Path(db_url.replace("duckdb://", "", 1))
    if not path.exists():
        return None
    import duckdb

    conn = duckdb.connect(str(path))
    try:
        result = conn.execute("SELECT COUNT(*) FROM reviews").fetchone()
        if result:
            return int(result[0])
    except duckdb.CatalogException:
        return None
    finally:
        conn.close()
    return None


def _log_step(step: StepResult) -> None:
    table = Table(title=f"Step: {step.name}")
    table.add_column("Metric")
    table.add_column("Value")
    for key, value in step.metrics.items():
        table.add_row(str(key), json.dumps(value) if isinstance(value, (list, dict)) else str(value))
    table.add_row("verifier", step.verifier_status)
    console.print(table)


def _record(results: List[StepResult], name: str, metrics: Dict[str, object], status: str) -> None:
    result = StepResult(name=name, metrics=metrics, verifier_status=status)
    results.append(result)
    _log_step(result)


def run_pipeline(config: RunConfig) -> int:
    results: List[StepResult] = []
    prev_count = _existing_review_count(config.db_url)

    # Step 1: fetch
    fetch_metrics = fetch.fetch_reviews(config.app_id, config.since, config.raw)
    if config.max_reviews is not None and fetch_metrics.get("rows", 0) > config.max_reviews:
        df = pd.read_csv(config.raw)
        df = df.head(config.max_reviews)
        df.to_csv(config.raw, index=False)
        fetch_metrics["rows"] = int(df.shape[0])
    if config.max_reviews is not None:
        ok, message = True, "test slice"
    else:
        ok, message = verifiers.verify_row_growth(prev_count, fetch_metrics.get("rows", 0))
    status = "pass" if ok else f"fail: {message}"
    _record(results, "fetch", fetch_metrics, status)
    if not ok:
        return 1

    # Step 2: prepare
    prepare_metrics = prepare.prepare(config.raw, config.clean)
    ok, message = verifiers.verify_lang_mix(pct_lang_kept=prepare_metrics.get("pct_lang_kept"))
    status = "pass" if ok else f"fail: {message}"
    _record(results, "prepare", prepare_metrics, status)
    if not ok:
        return 1

    # Step 3: dedupe with retry
    dedupe_key = "review_id|clean_text"
    dedupe_metrics = dedupe.dedupe(config.clean, config.unique, key=dedupe_key)
    ok, message = verifiers.verify_dup_rate(dedupe_metrics.get("dup_rate", 0.0))
    if not ok:
        console.print(f"[yellow]Dedupe verifier failed: {message}. Retrying with tighter key.[/yellow]")
        dedupe_key = fixers.tighten_dedupe_key(dedupe_key)
        dedupe_metrics = dedupe.dedupe(config.clean, config.unique, key=dedupe_key)
        ok, message = verifiers.verify_dup_rate(dedupe_metrics.get("dup_rate", 0.0))
    status = "pass" if ok else f"fail: {message}"
    _record(results, "dedupe", dedupe_metrics, status)
    if not ok:
        return 1

    # Step 4: embed
    embed_kwargs = {"model": config.embed_model}
    embed_metrics = embed.embed(config.unique, config.embedded, **embed_kwargs)
    ok = True
    message = "skipped"
    if config.embed_model != "none":
        ok, message = verifiers.verify_embed_coverage(embed_metrics.get("coverage", 0.0))
    status = "pass" if ok else f"fail: {message}"
    _record(results, "embed", embed_metrics, status)
    if not ok:
        return 1

    # Step 5: classify with retry
    classify_metrics = classify.classify(
        config.embedded,
        config.topics,
        taxonomy=config.taxonomy,
        min_conf=config.min_conf,
    )
    ok, message = verifiers.verify_topic_consistency(
        min_conf=config.min_conf,
        avg_conf=classify_metrics.get("avg_conf", 0.0),
        blank_pct=classify_metrics.get("blank_pct", 0.0),
    )
    if not ok:
        console.print(f"[yellow]Classification verifier failed: {message}. Lowering threshold.[/yellow]")
        config.min_conf = fixers.lower_topic_threshold(config.min_conf)
        classify_metrics = classify.classify(
            config.embedded,
            config.topics,
            taxonomy=config.taxonomy,
            min_conf=config.min_conf,
        )
        ok, message = verifiers.verify_topic_consistency(
            min_conf=config.min_conf,
            avg_conf=classify_metrics.get("avg_conf", 0.0),
            blank_pct=classify_metrics.get("blank_pct", 0.0),
        )
    status = "pass" if ok else f"fail: {message}"
    _record(results, "classify", classify_metrics, status)
    if not ok:
        return 1

    # Step 6: materialize
    materialize_metrics = materialize.load(
        config.db_url,
        reviews_parquet=config.embedded,
        topics_parquet=config.topics,
        use_pg=config.use_pg,
    )
    ok, message = verifiers.verify_views_materialized(materialize_metrics.get("tables", []))
    status = "pass" if ok else f"fail: {message}"
    _record(results, "materialize", materialize_metrics, status)
    if not ok:
        return 1

    # Step 7: report
    report_metrics = report.render(config.db_url, config.report, config.since, config.until)
    _record(results, "report", report_metrics, "pass")

    console.print("[green]Pipeline completed successfully.[/green]")
    return 0


__all__ = ["RunConfig", "run_pipeline"]
