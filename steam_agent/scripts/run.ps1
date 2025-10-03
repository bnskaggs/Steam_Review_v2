param(
    [int]$AppId = 1364780,
    [string]$Since = "2025-07-01T00:00:00+00:00",
    [string]$Until = "2025-10-01T00:00:00+00:00"
)

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$project = Join-Path $root ".."

python -m steam_agent.cli run `
    --app-id $AppId `
    --since $Since `
    --until $Until `
    --raw (Join-Path $project "data/raw.csv") `
    --clean (Join-Path $project "data/clean.parquet") `
    --unique (Join-Path $project "data/unique.parquet") `
    --embedded (Join-Path $project "data/embedded.parquet") `
    --topics (Join-Path $project "data/topics.parquet") `
    --db ("duckdb://" + (Join-Path $project "db/reviews.duckdb")) `
    --report (Join-Path $project "data/weekly.md")
