# Steam Review Agent

A deterministic, testable Steam review processing pipeline operated by a tiny agent loop. The system fetches reviews, cleans and deduplicates them, applies rule-first topic classifications, materializes the results into DuckDB, and renders a weekly Markdown digest. Optional hooks exist for embeddings and Postgres/LLM integrations without changing the default workflow.

## Features

- **Deterministic batch pipeline** using explicit file inputs/outputs (CSV/Parquet).
- **Agentic control loop** orchestrating steps, running verifiers, and applying targeted fixers for dedupe and classification retries.
- **Rich logging** via `rich` for readable step summaries.
- **Windows-friendly tooling** with a `run.ps1` helper and pure-Python dependencies.
- **Extensible architecture** – optional embedding model, Postgres/pgvector paths, and future LLM classifiers are gated behind flags.

## Quick start

1. Create a Python 3.10+ virtual environment and install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
   pip install -e .
   ```

2. Run the small validation slice (fetches ~100 reviews for the last 30 days):

   ```bash
   python -m steam_agent.cli test-slice
   ```

3. Execute a full run (example for *TEKKEN 8*, app id `1364780`):

   ```bash
   python -m steam_agent.cli run \
     --app-id 1364780 \
     --since 2025-07-01T00:00:00+00:00 \
     --until 2025-10-01T00:00:00+00:00 \
     --raw steam_agent/data/raw.csv \
     --clean steam_agent/data/clean.parquet \
     --unique steam_agent/data/unique.parquet \
     --embedded steam_agent/data/embedded.parquet \
     --topics steam_agent/data/topics.parquet \
     --db duckdb://steam_agent/db/reviews.duckdb \
     --report steam_agent/data/weekly.md
   ```

4. Review outputs:

   - `steam_agent/data/weekly.md` – Markdown summary with daily volumes and top topics.
   - `steam_agent/db/reviews.duckdb` – DuckDB database with `reviews` and `review_topics` tables.

## Tests

Run the unit tests to validate verifiers and rule-based classification logic:

```bash
pytest
```

## PowerShell helper

On Windows, use the included script:

```powershell
./steam_agent/scripts/run.ps1
```

It forwards recommended arguments to the CLI for the canonical TEKKEN 8 pipeline run.

## Extending the pipeline

- **Embedding**: Switch `--embed-model` away from `none` and supply an embedder inside `pipeline/embed.py`. The verifier already enforces coverage.
- **Topic taxonomy**: Edit `steam_agent/agent/taxonomy.yaml` to update keyword mappings.
- **Postgres**: Set `--use-pg` with a `postgresql://` `--db` URL once psycopg/pgvector are installed. The stub emits a warning until implemented.

