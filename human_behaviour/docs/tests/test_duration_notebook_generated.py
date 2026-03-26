from src.scoring.leader_scorer import clamp01


def test_clamp01_bounds():
    assert clamp01(-1.5) == 0.0
    assert clamp01(0.5) == 0.5
    assert clamp01(3.0) == 1.0
