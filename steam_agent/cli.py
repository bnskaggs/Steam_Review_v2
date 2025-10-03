"""Command-line interface for the Steam review pipeline."""
from __future__ import annotations

import argparse
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .agent.loop import RunConfig, run_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


def _default_paths(root: Path) -> dict:
    return {
        "raw": str(root / "data" / "raw.csv"),
        "clean": str(root / "data" / "clean.parquet"),
        "unique": str(root / "data" / "unique.parquet"),
        "embedded": str(root / "data" / "embedded.parquet"),
        "topics": str(root / "data" / "topics.parquet"),
        "report": str(root / "data" / "weekly.md"),
        "db": f"duckdb://{root / 'db' / 'reviews.duckdb'}",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Steam review pipeline operator")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the full pipeline")
    run_parser.add_argument("--app-id", type=int, required=True)
    run_parser.add_argument("--since", type=str, required=True)
    run_parser.add_argument("--until", type=str, required=True)
    run_parser.add_argument("--raw", type=str, required=True)
    run_parser.add_argument("--clean", type=str, required=True)
    run_parser.add_argument("--unique", type=str, required=True)
    run_parser.add_argument("--embedded", type=str, required=True)
    run_parser.add_argument("--topics", type=str, required=True)
    run_parser.add_argument("--db", type=str, required=True, help="Database URL")
    run_parser.add_argument("--report", type=str, required=True)
    run_parser.add_argument("--taxonomy", type=str, default=str(Path(__file__).parent / "agent" / "taxonomy.yaml"))
    run_parser.add_argument("--embed-model", type=str, default="none")
    run_parser.add_argument("--min-conf", type=float, default=0.55)
    run_parser.add_argument("--use-pg", action="store_true")

    test_parser = subparsers.add_parser("test-slice", help="Run a small validation slice")
    test_parser.add_argument("--app-id", type=int, default=1364780)
    test_parser.add_argument("--days", type=int, default=30)
    test_parser.add_argument("--limit", type=int, default=100)

    return parser


def _parse_run_args(args: argparse.Namespace) -> RunConfig:
    return RunConfig(
        app_id=args.app_id,
        since=args.since,
        until=args.until,
        raw=args.raw,
        clean=args.clean,
        unique=args.unique,
        embedded=args.embedded,
        topics=args.topics,
        db_url=args.db,
        report=args.report,
        taxonomy=args.taxonomy,
        embed_model=args.embed_model,
        min_conf=args.min_conf,
        use_pg=args.use_pg,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        config = _parse_run_args(args)
    else:
        root = Path(__file__).resolve().parent
        defaults = _default_paths(root)
        until = datetime.now(tz=timezone.utc)
        since = until - timedelta(days=args.days)
        config = RunConfig(
            app_id=args.app_id,
            since=since.isoformat(),
            until=until.isoformat(),
            raw=defaults["raw"],
            clean=defaults["clean"],
            unique=defaults["unique"],
            embedded=defaults["embedded"],
            topics=defaults["topics"],
            db_url=defaults["db"],
            report=defaults["report"],
            taxonomy=str(Path(__file__).resolve().parent / "agent" / "taxonomy.yaml"),
            max_reviews=args.limit,
        )

    return run_pipeline(config)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
