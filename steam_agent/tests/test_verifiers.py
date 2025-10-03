from steam_agent.agent import verifiers


def test_verify_row_growth():
    assert verifiers.verify_row_growth(None, 10)
    assert verifiers.verify_row_growth(100, 105, min_growth=0.04)
    assert not verifiers.verify_row_growth(100, 101, min_growth=0.05)
    assert not verifiers.verify_row_growth(100, 0)


def test_verify_dup_rate():
    assert verifiers.verify_dup_rate(0.01)
    assert verifiers.verify_dup_rate(0.03, max_dup=0.05)
    assert not verifiers.verify_dup_rate(0.2)
