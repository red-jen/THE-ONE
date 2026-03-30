from src.scoring.leader_scorer import ScoringWeights, aggregate, score_person


def test_aggregate_and_score_person():
    rows = [
        {
            "person_id": 1,
            "camera_id": "cam_a",
            "timestamp_sec": 0.0,
            "detection_confidence": 0.9,
            "signals": {
                "position_hint": "front",
                "has_megaphone": True,
                "has_banner": False,
                "has_flag": False,
                "has_microphone": True,
                "gesture_score": 2,
            },
        },
        {
            "person_id": 1,
            "camera_id": "cam_b",
            "timestamp_sec": 20.0,
            "detection_confidence": 0.8,
            "signals": {
                "position_hint": "front",
                "has_megaphone": True,
                "has_banner": True,
                "has_flag": False,
                "has_microphone": False,
                "gesture_score": 1,
            },
        },
    ]

    grouped = aggregate(rows)
    assert 1 in grouped

    scored = score_person(grouped[1], ScoringWeights())
    assert scored["person_id"] == 1
    assert scored["suspicion_score"] > 0
    assert scored["components"]["front_presence"] > 0
    assert scored["components"]["object_signal"] > 0
    assert scored["stats"]["camera_count"] == 2
