"""Pipeline operator control loop."""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from tenacity import RetryError, retry, stop_after_attempt, wait_exponential

from ..pipeline import classify, dedupe, embed, fetch, materialize, prepare, report
from . import fixers, verifiers

LOGGER = logging.getLogger(__name__)
console = Console()


@dataclass
class RunConfig:
    """Arguments required to execute the pipeline once."""

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
    taxonomy: str = str(Path(__file__).with_name("taxonomy.yaml"))
    embed_model: str = "none"
    min_conf: float = 0.55
    use_pg: bool = False
    max_reviews: Optional[int] = None
    langs: Optional[List[str]] = None


def _existing_review_count(db_url: str) -> Optional[int]:
    if not db_url.startswith("duckdb://"):
        return None
    path = Path(db_url.replace("duckdb://", "", 1))
    if not path.exists():
        return None
    import duckdb  # Imported lazily to keep dependency optional.

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


@retry(reraise=True, wait=wait_exponential(multiplier=1, min=1, max=8), stop=stop_after_attempt(3))
def _fetch_with_retry(config: RunConfig) -> Dict[str, object]:
    return fetch.fetch_reviews(config.app_id, config.since, config.raw)


def _log_banner(config: RunConfig) -> None:
    panel = Panel(
        f"App ID: [bold]{config.app_id}[/bold]\n"
        f"Window: [cyan]{config.since}[/cyan] → [cyan]{config.until}[/cyan]",
        title="Steam Review Agent",
        style="bold blue",
    )
    console.print(panel)


def _log_step(name: str, metrics: Dict[str, object], attempt: int, success: bool) -> None:
    table = Table(title=f"{name} (attempt {attempt})", title_style="bold")
    table.add_column("Metric", style="dim")
    table.add_column("Value")
    for key, value in metrics.items():
        pretty = json.dumps(value) if isinstance(value, (dict, list)) else str(value)
        table.add_row(str(key), pretty)
    status = "✅ pass" if success else "❌ fail"
    table.add_row("status", status)
    console.print(table)


StepRunner = Callable[[], Dict[str, object]]
Verifier = Optional[Callable[[Dict[str, object]], bool]]
Fixer = Optional[Callable[[Dict[str, object]], None]]


def _run_step(name: str, runner: StepRunner, verifier: Verifier, fixer: Fixer) -> tuple[Dict[str, object], bool]:
    attempt = 1
    metrics = runner()
    success = True if verifier is None else verifier(metrics)
    _log_step(name, metrics, attempt, success)
    if success:
        return metrics, True

    if fixer is None:
        return metrics, False

    fixer(metrics)
    attempt += 1
    metrics = runner()
    success = True if verifier is None else verifier(metrics)
    _log_step(name, metrics, attempt, success)
    return metrics, success


def _log_summary(results: List[tuple[str, Dict[str, object]]]) -> None:
    summary = Table(title="Pipeline Summary", title_style="bold green")
    summary.add_column("Step", style="bold")
    summary.add_column("Metric")
    summary.add_column("Value")
    for name, metrics in results:
        for idx, (key, value) in enumerate(metrics.items()):
            metric_name = name if idx == 0 else ""
            pretty = json.dumps(value) if isinstance(value, (dict, list)) else str(value)
            summary.add_row(metric_name, str(key), pretty)
    console.print(summary)


def run_pipeline(config: RunConfig) -> int:
    _log_banner(config)
    results: List[tuple[str, Dict[str, object]]] = []
    previous_rows = _existing_review_count(config.db_url)

    try:
        fetch_metrics = _fetch_with_retry(config)
    except RetryError as exc:  # pragma: no cover - network error path
        LOGGER.error("Fetch failed after retries: %s", exc)
        return 1
    except Exception as exc:  # pragma: no cover - unexpected failure
        LOGGER.exception("Fetch crashed: %s", exc)
        return 1

    if config.max_reviews is not None:
        import pandas as pd

        df = pd.read_csv(config.raw).head(config.max_reviews)
        df.to_csv(config.raw, index=False)
        fetch_metrics["rows"] = int(df.shape[0])
        fetch_ok = True
    else:
        fetch_ok = verifiers.verify_row_growth(previous_rows, int(fetch_metrics.get("rows", 0)))
    _log_step("fetch", fetch_metrics, 1, fetch_ok)
    results.append(("fetch", fetch_metrics))
    if not fetch_ok:
        LOGGER.error("Row growth verification failed (prev=%s, new=%s)", previous_rows, fetch_metrics.get("rows"))
        _log_summary(results)
        return 1

    prepare_metrics = prepare.prepare(config.raw, config.clean, langs=config.langs)
    _log_step("prepare", prepare_metrics, 1, True)
    results.append(("prepare", prepare_metrics))

    lang_kept = float(prepare_metrics.get("pct_lang_kept", 0.0))
    lang_counts_kept = prepare_metrics.get("lang_counts_kept") or {}
    if not isinstance(lang_counts_kept, dict):
        lang_counts_kept = dict(lang_counts_kept)
    total_rows_kept = int(prepare_metrics.get("rows_in_kept") or sum(lang_counts_kept.values()) or 0)

    dedupe_key = "review_id|clean_text"

    def run_dedupe() -> Dict[str, object]:
        return dedupe.dedupe(config.clean, config.unique, key=dedupe_key)

    def fix_dedupe(_: Dict[str, object]) -> None:
        nonlocal dedupe_key
        dedupe_key = fixers.tighten_dedupe_key(dedupe_key)

    dedupe_metrics, dedupe_ok = _run_step(
        "dedupe",
        run_dedupe,
        lambda m: verifiers.verify_dup_rate(float(m.get("dup_rate", 0.0))),
        fix_dedupe,
    )
    results.append(("dedupe", dedupe_metrics))
    if not dedupe_ok:
        LOGGER.error("Dedupe verification failed twice; aborting.")
        _log_summary(results)
        return 1

    embed_force = False

    def run_embed() -> Dict[str, object]:
        return embed.embed(config.unique, config.embedded, model=config.embed_model, force=embed_force)

    def fix_embed(_: Dict[str, object]) -> None:
        nonlocal embed_force
        embed_force = fixers.force_reembed()["force"]

    def embed_verify(metrics: Dict[str, object]) -> bool:
        rows = int(metrics.get("rows_in", 0))
        if rows == 0:
            return True
        return float(metrics.get("coverage", 0.0)) >= 0.99

    embed_metrics, embed_ok = _run_step("embed", run_embed, embed_verify, fix_embed)
    results.append(("embed", embed_metrics))
    if not embed_ok:
        LOGGER.error("Embedding coverage below threshold after retry.")
        _log_summary(results)
        return 1

    classify_attempt = 1

    def _run_classify_attempt() -> tuple[Dict[str, object], bool, dict[str, float | bool]]:
        metrics = classify.classify(
            config.embedded,
            config.topics,
            taxonomy=config.taxonomy,
            min_conf=config.min_conf,
        )
        blank_pct = float(metrics.get("blank_pct", 1.0))
        actual_labeled = max(0.0, min(1.0, 1.0 - blank_pct))
        metrics["actual_labeled_pct"] = round(actual_labeled, 4)
        metrics["overall_topic_rate"] = round(lang_kept * actual_labeled, 4)
        metrics["lang_kept"] = round(lang_kept, 4)
        ok, info = verifiers.verify_topic_density(
            blank_pct=blank_pct,
            lang_counts=lang_counts_kept,
            supported_langs={"en"},
            total_rows=total_rows_kept,
        )
        metrics["expected_overall"] = round(info["expected_overall"], 4)
        metrics["supported_share"] = round(info["supported_share"], 4)
        metrics["strict_density_check"] = bool(info["strict"])
        return metrics, ok, info

    classify_metrics, blank_ok, density_info = _run_classify_attempt()
    _log_step("classify", classify_metrics, classify_attempt, blank_ok)

    final_density_info = density_info

    if blank_ok:
        results.append(("classify", classify_metrics))
    else:
        new_min_conf = max(0.35, config.min_conf - 0.05)
        if new_min_conf != config.min_conf:
            LOGGER.warning(
                "Topic density below threshold (blank_pct=%s); retrying classify with min_conf=%s",
                classify_metrics.get("blank_pct"),
                new_min_conf,
            )
        else:
            LOGGER.warning(
                "Topic density below threshold (blank_pct=%s); retrying classify at floor min_conf=%s",
                classify_metrics.get("blank_pct"),
                new_min_conf,
            )
        config.min_conf = new_min_conf
        classify_attempt += 1
        classify_metrics, blank_ok, density_info = _run_classify_attempt()
        final_density_info = density_info

        final_ok = blank_ok
        if not final_ok:
            supported_share = float(density_info["supported_share"])
            strict = bool(density_info["strict"])
            if supported_share < 0.15 or not strict:
                LOGGER.warning(
                    "Topic density below expectation but tolerated (supported_share=%.2f, strict=%s)",
                    supported_share,
                    strict,
                )
                final_ok = True
            else:
                LOGGER.error(
                    "Topic density verification failed (actual=%.2f, expected=%.2f, supported_share=%.2f)",
                    density_info["actual_labeled"],
                    density_info["expected_overall"],
                    supported_share,
                )
        _log_step("classify", classify_metrics, classify_attempt, final_ok)
        results.append(("classify", classify_metrics))
        if not final_ok:
            _log_summary(results)
            return 1

    materialize_metrics = materialize.load(
        config.db_url,
        reviews_parquet=config.embedded,
        topics_parquet=config.topics,
        use_pg=config.use_pg,
    )
    _log_step("materialize", materialize_metrics, 1, True)
    results.append(("materialize", materialize_metrics))

    report_metrics = report.render(
        config.db_url,
        config.report,
        config.since,
        config.until,
        lang_counts=lang_counts_kept,
        actual_labeled=float(final_density_info["actual_labeled"]),
        expected_overall=float(final_density_info["expected_overall"]),
        supported_share=float(final_density_info["supported_share"]),
        lang_kept=lang_kept,
    )
    _log_step("report", report_metrics, 1, True)
    results.append(("report", report_metrics))

    _log_summary(results)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Steam review pipeline loop")
    parser.add_argument("--app-id", type=int, required=True)
    parser.add_argument("--since", type=str, required=True)
    parser.add_argument("--until", type=str, required=True)
    parser.add_argument("--raw", type=str, required=True)
    parser.add_argument("--clean", type=str, required=True)
    parser.add_argument("--unique", type=str, required=True)
    parser.add_argument("--embedded", type=str, required=True)
    parser.add_argument("--topics", type=str, required=True)
    parser.add_argument("--db-url", type=str, required=True)
    parser.add_argument("--report", type=str, required=True)
    parser.add_argument("--taxonomy", type=str, default=str(Path(__file__).with_name("taxonomy.yaml")))
    parser.add_argument("--embed-model", type=str, default="none")
    parser.add_argument("--min-conf", type=float, default=0.55)
    parser.add_argument("--use-pg", action="store_true")
    parser.add_argument("--max-reviews", type=int)
    parser.add_argument("--langs", type=str, default="en", help="Comma-separated language codes")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    config = RunConfig(
        app_id=args.app_id,
        since=args.since,
        until=args.until,
        raw=args.raw,
        clean=args.clean,
        unique=args.unique,
        embedded=args.embedded,
        topics=args.topics,
        db_url=args.db_url,
        report=args.report,
        taxonomy=args.taxonomy,
        embed_model=args.embed_model,
        min_conf=args.min_conf,
        use_pg=args.use_pg,
        max_reviews=args.max_reviews,
        langs=[code.strip().lower() for code in args.langs.split(",") if code.strip()],
    )
    return run_pipeline(config)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["RunConfig", "run_pipeline", "main"]
