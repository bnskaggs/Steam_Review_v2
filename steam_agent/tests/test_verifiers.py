from steam_agent.agent import verifiers


def test_verify_row_growth_pass():
    ok, msg = verifiers.verify_row_growth(100, 120, 0.1)
    assert ok
    assert "row growth" in msg


def test_verify_row_growth_fail():
    ok, msg = verifiers.verify_row_growth(100, 105, 0.1)
    assert not ok
    assert "below" in msg


def test_verify_lang_mix():
    ok, _ = verifiers.verify_lang_mix(pct_lang_kept=0.95)
    assert ok
    ok, _ = verifiers.verify_lang_mix(pct_lang_kept=0.5)
    assert not ok


def test_verify_dup_rate():
    ok, _ = verifiers.verify_dup_rate(0.01)
    assert ok
    ok, msg = verifiers.verify_dup_rate(0.5)
    assert not ok
    assert "exceeds" in msg


def test_verify_topic_consistency():
    ok, _ = verifiers.verify_topic_consistency(0.5, 0.6, 0.05)
    assert ok
    ok, _ = verifiers.verify_topic_consistency(0.7, 0.6, 0.05)
    assert not ok


def test_verify_views_materialized():
    ok, _ = verifiers.verify_views_materialized(["reviews", "review_topics"])
    assert ok
    ok, msg = verifiers.verify_views_materialized(["reviews"])
    assert not ok
    assert "missing" in msg
