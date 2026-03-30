from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class PersonAggregate:
    person_id: int
    observations: int = 0
    front_count: int = 0
    center_count: int = 0
    back_count: int = 0
    unknown_pos_count: int = 0
    megaphone_count: int = 0
    banner_count: int = 0
    flag_count: int = 0
    microphone_count: int = 0
    gesture_total: int = 0
    confidence_sum: float = 0.0
    min_ts: float = float("inf")
    max_ts: float = 0.0
    cameras: set[str] | None = None

    def __post_init__(self) -> None:
        if self.cameras is None:
            self.cameras = set()


@dataclass
class ScoringWeights:
    front_presence: float = 25.0
    object_signal: float = 25.0
    gesture_signal: float = 15.0
    duration_signal: float = 10.0
    multi_camera_signal: float = 10.0
    presence_signal: float = 15.0


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def load_description_rows(path: str) -> List[dict]:
    rows: List[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def aggregate(rows: List[dict]) -> Dict[int, PersonAggregate]:
    agg: Dict[int, PersonAggregate] = {}

    for row in rows:
        person_id = int(row["person_id"])
        signals = row.get("signals", {})
        camera_id = str(row.get("camera_id", "unknown"))
        ts = float(row.get("timestamp_sec", 0.0))
        confidence = float(row.get("detection_confidence", 0.0))

        item = agg.get(person_id)
        if item is None:
            item = PersonAggregate(person_id=person_id)
            agg[person_id] = item

        item.observations += 1
        item.confidence_sum += confidence
        item.min_ts = min(item.min_ts, ts)
        item.max_ts = max(item.max_ts, ts)
        item.cameras.add(camera_id)

        position = str(signals.get("position_hint", "unknown"))
        if position == "front":
            item.front_count += 1
        elif position == "center":
            item.center_count += 1
        elif position == "back":
            item.back_count += 1
        else:
            item.unknown_pos_count += 1

        item.megaphone_count += int(bool(signals.get("has_megaphone", False)))
        item.banner_count += int(bool(signals.get("has_banner", False)))
        item.flag_count += int(bool(signals.get("has_flag", False)))
        item.microphone_count += int(bool(signals.get("has_microphone", False)))
        item.gesture_total += int(signals.get("gesture_score", 0))

    return agg


def compute_object_score(item: PersonAggregate) -> float:
    obs = max(1, item.observations)

    megaphone_ratio = item.megaphone_count / obs
    banner_ratio = item.banner_count / obs
    flag_ratio = item.flag_count / obs
    microphone_ratio = item.microphone_count / obs

    weighted = (
        0.40 * megaphone_ratio
        + 0.25 * microphone_ratio
        + 0.20 * banner_ratio
        + 0.15 * flag_ratio
    )
    return clamp01(weighted)


def score_person(item: PersonAggregate, weights: ScoringWeights) -> dict:
    obs = max(1, item.observations)

    front_ratio = item.front_count / obs
    object_ratio = compute_object_score(item)
    gesture_ratio = clamp01(item.gesture_total / (obs * 2.0))

    duration_sec = max(0.0, item.max_ts - (item.min_ts if item.min_ts != float("inf") else 0.0))
    duration_ratio = clamp01(duration_sec / 30.0)

    camera_count = len(item.cameras)
    multi_camera_ratio = clamp01((camera_count - 1) / 2.0)

    presence_ratio = clamp01(item.observations / 20.0)

    components = {
        "front_presence": weights.front_presence * front_ratio,
        "object_signal": weights.object_signal * object_ratio,
        "gesture_signal": weights.gesture_signal * gesture_ratio,
        "duration_signal": weights.duration_signal * duration_ratio,
        "multi_camera_signal": weights.multi_camera_signal * multi_camera_ratio,
        "presence_signal": weights.presence_signal * presence_ratio,
    }

    suspicion_score = sum(components.values())

    return {
        "person_id": item.person_id,
        "suspicion_score": round(float(suspicion_score), 3),
        "leader_score": round(float(suspicion_score), 3),
        "components": {k: round(float(v), 3) for k, v in components.items()},
        "stats": {
            "observations": item.observations,
            "avg_detection_confidence": round(item.confidence_sum / obs, 6),
            "duration_sec": round(duration_sec, 3),
            "camera_count": camera_count,
            "front_count": item.front_count,
            "center_count": item.center_count,
            "back_count": item.back_count,
            "unknown_pos_count": item.unknown_pos_count,
            "megaphone_count": item.megaphone_count,
            "banner_count": item.banner_count,
            "flag_count": item.flag_count,
            "microphone_count": item.microphone_count,
            "gesture_total": item.gesture_total,
        },
    }


def run_leader_scoring(
    input_jsonl: str,
    output_jsonl: str,
    output_summary: str,
    top_k: int,
    use_mlflow: bool,
    mlflow_tracking_uri: str | None,
    mlflow_experiment: str,
    mlflow_run_name: str | None,
) -> None:
    mlflow = None
    if use_mlflow:
        try:
            mlflow = importlib.import_module("mlflow")
            if mlflow_tracking_uri:
                mlflow.set_tracking_uri(mlflow_tracking_uri)
            mlflow.set_experiment(mlflow_experiment)
            mlflow.start_run(run_name=mlflow_run_name)
            mlflow.log_param("input_jsonl", str(Path(input_jsonl).resolve()))
            mlflow.log_param("top_k", top_k)
        except Exception as exc:
            print(f"[WARN] MLflow requested but unavailable: {exc}")
            mlflow = None

    rows = load_description_rows(input_jsonl)
    if not rows:
        raise RuntimeError(f"No description rows found in {input_jsonl}")

    agg = aggregate(rows)
    weights = ScoringWeights()

    scored = [score_person(item, weights=weights) for item in agg.values()]
    scored.sort(key=lambda x: x["suspicion_score"], reverse=True)

    out_jsonl_path = Path(output_jsonl).resolve()
    out_summary_path = Path(output_summary).resolve()
    out_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    out_summary_path.parent.mkdir(parents=True, exist_ok=True)

    with out_jsonl_path.open("w", encoding="utf-8") as f:
        for row in scored:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    top = scored[: max(1, top_k)]
    summary = {
        "input_jsonl": str(Path(input_jsonl).resolve()),
        "people_scored": len(scored),
        "top_k": top_k,
        "top_candidates": top,
        "weights": {
            "front_presence": weights.front_presence,
            "object_signal": weights.object_signal,
            "gesture_signal": weights.gesture_signal,
            "duration_signal": weights.duration_signal,
            "multi_camera_signal": weights.multi_camera_signal,
            "presence_signal": weights.presence_signal,
        },
    }

    with out_summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if mlflow is not None:
        mlflow.log_param("weights.front_presence", weights.front_presence)
        mlflow.log_param("weights.object_signal", weights.object_signal)
        mlflow.log_param("weights.gesture_signal", weights.gesture_signal)
        mlflow.log_param("weights.duration_signal", weights.duration_signal)
        mlflow.log_param("weights.multi_camera_signal", weights.multi_camera_signal)
        mlflow.log_param("weights.presence_signal", weights.presence_signal)

        mlflow.log_metric("people_scored", len(scored))
        mlflow.log_metric("rows_input", len(rows))
        if scored:
            mlflow.log_metric("top_suspicion_score", float(scored[0]["suspicion_score"]))

        mlflow.log_artifact(str(out_jsonl_path))
        mlflow.log_artifact(str(out_summary_path))
        mlflow.end_run()

    print("=" * 60)
    print("Suspicion scoring completed")
    print(f"Input: {Path(input_jsonl).resolve()}")
    print(f"Scores JSONL: {out_jsonl_path}")
    print(f"Summary JSON: {out_summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score tracked persons by protest suspicion signals")
    parser.add_argument("--input-jsonl", type=str, required=True, help="Description events JSONL path")
    parser.add_argument("--output-jsonl", type=str, default="outputs/scores/suspicion_scores.jsonl")
    parser.add_argument("--output-summary", type=str, default="outputs/scores/suspicion_scores_summary.json")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top candidates in summary")
    parser.add_argument("--mlflow", action="store_true", help="Enable MLflow logging")
    parser.add_argument("--mlflow-tracking-uri", type=str, default=None, help="Optional MLflow tracking URI")
    parser.add_argument("--mlflow-experiment", type=str, default="human_behaviour_scoring")
    parser.add_argument("--mlflow-run-name", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_leader_scoring(
        input_jsonl=args.input_jsonl,
        output_jsonl=args.output_jsonl,
        output_summary=args.output_summary,
        top_k=args.top_k,
        use_mlflow=args.mlflow,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        mlflow_experiment=args.mlflow_experiment,
        mlflow_run_name=args.mlflow_run_name,
    )


if __name__ == "__main__":
    main()
