from src.memory.chroma_store import build_simple_store, query_simple_store


def test_simple_store_query_returns_ranked_results():
    rows = [
        {
            "person_id": 10,
            "camera_id": "cam1",
            "frame_idx": 1,
            "timestamp_sec": 0.0,
            "bbox": [0, 0, 10, 10],
            "description": "Person at front holding megaphone and speaking to crowd",
            "signals": {
                "has_megaphone": True,
                "has_banner": False,
                "has_flag": False,
                "has_microphone": False,
                "gesture_score": 2,
                "position_hint": "front",
            },
        },
        {
            "person_id": 11,
            "camera_id": "cam1",
            "frame_idx": 2,
            "timestamp_sec": 1.0,
            "bbox": [0, 0, 10, 10],
            "description": "Person walking at rear with no visible object",
            "signals": {
                "has_megaphone": False,
                "has_banner": False,
                "has_flag": False,
                "has_microphone": False,
                "gesture_score": 0,
                "position_hint": "back",
            },
        },
    ]

    store = build_simple_store(rows)
    results = query_simple_store(store, query="leader with megaphone at front", top_k=2)

    assert len(results) >= 1
    assert results[0]["metadata"]["person_id"] == 10
    assert results[0]["score"] >= 0
