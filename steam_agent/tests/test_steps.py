import pandas as pd
import pytest

from steam_agent.agent import fixers
from steam_agent.pipeline import classify, dedupe


def test_dedupe_tightens_key(tmp_path):
    data = pd.DataFrame(
        [
            {
                "review_id": "1",
                "clean_text": "great game",
                "ts": "2024-01-01",
                "lang": "english",
                "version_checksum": "a",
            },
            {
                "review_id": "1",
                "clean_text": "great game",
                "ts": "2024-01-01",
                "lang": "english",
                "version_checksum": "a",
            },
            {
                "review_id": "2",
                "clean_text": "great game",
                "ts": "2024-01-01",
                "lang": "english",
                "version_checksum": "b",
            },
        ]
    )
    src = tmp_path / "clean.parquet"
    out = tmp_path / "unique.parquet"
    data.to_parquet(src, index=False)

    metrics = dedupe.dedupe(str(src), str(out))
    assert metrics["rows_out"] == 2

    tighter_key = fixers.tighten_dedupe_key("review_id|clean_text")
    metrics_tight = dedupe.dedupe(str(src), str(out), key=tighter_key)
    assert metrics_tight["rows_out"] == 2
    assert metrics_tight["dup_rate"] <= metrics["dup_rate"]


def test_classify_rules(tmp_path):
    reviews = pd.DataFrame(
        [
            {
                "review_id": "1",
                "app_id": 1,
                "ts": "2024-01-01T00:00:00+00:00",
                "lang": "english",
                "clean_text": "The combat feels great and combos are awesome",
                "helpful": 0,
                "funny": 0,
                "version_checksum": "abc",
                "embed_model": "none",
            },
            {
                "review_id": "2",
                "app_id": 1,
                "ts": "2024-01-01T00:00:00+00:00",
                "lang": "english",
                "clean_text": "The price is terrible and the paywall is bad",
                "helpful": 0,
                "funny": 0,
                "version_checksum": "def",
                "embed_model": "none",
            },
        ]
    )
    src = tmp_path / "embedded.parquet"
    out = tmp_path / "topics.parquet"
    reviews.to_parquet(src, index=False)

    taxonomy = tmp_path / "taxonomy.yaml"
    taxonomy.write_text(
        """
        topics:
          combat: ["combat", "combo"]
          monetization: ["price", "paywall"]
        """
    )

    metrics = classify.classify(str(src), str(out), taxonomy=str(taxonomy), min_conf=0.5)
    assert metrics["rows_labeled"] >= 2

    labeled = pd.read_parquet(out)
    assert set(labeled["topic"]) == {"combat", "monetization"}
