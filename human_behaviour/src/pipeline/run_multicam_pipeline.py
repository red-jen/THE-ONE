from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import sys
from typing import Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from description.llava_descriptor import run_description_pipeline
from interface.chatbot import answer_query
from memory.chroma_store import run_index
from scoring.leader_scorer import run_leader_scoring
from scoring.rag_scorer import run_rag_scoring
from tracking.reid_tracker import run_tracking


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def _sanitize(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", text.strip())
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned.strip("_") or "camera"


def _collect_videos(videos: List[str], videos_dir: str | None) -> List[Path]:
    collected: List[Path] = []

    for raw in videos:
        p = Path(raw).resolve()
        if p.exists() and p.is_file():
            collected.append(p)

    if videos_dir:
        base = Path(videos_dir).resolve()
        if base.exists() and base.is_dir():
            for path in sorted(base.iterdir()):
                if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
                    collected.append(path.resolve())

    uniq = []
    seen = set()
    for p in collected:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)

    return uniq


def _merge_jsonl(inputs: Iterable[Path], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as out_f:
        for src in inputs:
            with src.open("r", encoding="utf-8") as in_f:
                for line in in_f:
                    if not line.strip():
                        continue
                    out_f.write(line)
                    count += 1
    return count


def run_multicam_pipeline(
    videos: List[str],
    videos_dir: str | None,
    run_name: str,
    output_root: str,
    yolo_model: str,
    osnet_ckpt: str,
    sim_threshold: float,
    max_age: int,
    det_conf: float,
    det_iou: float,
    device: str,
    descriptor_backend: str,
    llava_model_id: str,
    strict_llava: bool,
    sample_every: int,
    max_events: int | None,
    score_top_k: int,
    memory_backend: str,
    question: str | None,
    top_leaders: int,
    top_evidence: int,
    rag_generator_backend: str,
    rag_ollama_model: str,
    rag_ollama_url: str,
    rag_ollama_timeout_sec: float,
) -> None:
    video_paths = _collect_videos(videos=videos, videos_dir=videos_dir)
    if not video_paths:
        raise RuntimeError("No input videos found. Provide --videos and/or --videos-dir with valid video files.")

    root = Path(output_root).resolve() / _sanitize(run_name)
    root.mkdir(parents=True, exist_ok=True)

    description_files: List[Path] = []
    camera_runs: List[dict] = []

    for idx, video_path in enumerate(video_paths, start=1):
        camera_id = f"cam_{idx}_{_sanitize(video_path.stem)}"
        cam_dir = root / camera_id
        track_dir = cam_dir / "tracking"
        desc_dir = cam_dir / "description"

        run_tracking(
            source=str(video_path),
            yolo_model=yolo_model,
            osnet_ckpt=osnet_ckpt,
            output_dir=str(track_dir),
            sim_threshold=sim_threshold,
            max_age=max_age,
            det_conf=det_conf,
            det_iou=det_iou,
            device=device,
        )

        tracks_jsonl = track_dir / "tracks.jsonl"
        desc_jsonl = desc_dir / "descriptions.jsonl"
        desc_summary = desc_dir / "summary.json"

        run_description_pipeline(
            source=str(video_path),
            tracks_jsonl=str(tracks_jsonl),
            output_jsonl=str(desc_jsonl),
            output_summary=str(desc_summary),
            camera_id=camera_id,
            prompt=(
                "Describe this person's actions, position in the crowd, visible objects, and body language. "
                "If uncertain, state uncertainty explicitly."
            ),
            backend_name=descriptor_backend,
            model_id=llava_model_id,
            device=device,
            max_new_tokens=96,
            sample_every=sample_every,
            max_events=max_events,
            strict_llava=strict_llava,
        )

        description_files.append(desc_jsonl)
        camera_runs.append(
            {
                "camera_id": camera_id,
                "video": str(video_path),
                "tracking_dir": str(track_dir),
                "tracks_jsonl": str(tracks_jsonl),
                "description_jsonl": str(desc_jsonl),
                "description_summary": str(desc_summary),
            }
        )

    merged_desc = root / "merged_descriptions.jsonl"
    merged_count = _merge_jsonl(description_files, merged_desc)

    # Stage 4a: heuristic pre-scoring (signal aggregation baseline)
    heuristic_jsonl = root / "heuristic_scores.jsonl"
    heuristic_summary = root / "heuristic_scores_summary.json"
    run_leader_scoring(
        input_jsonl=str(merged_desc),
        output_jsonl=str(heuristic_jsonl),
        output_summary=str(heuristic_summary),
        top_k=score_top_k,
        use_mlflow=False,
        mlflow_tracking_uri=None,
        mlflow_experiment="human_behaviour_scoring",
        mlflow_run_name=None,
    )

    # Stage 4b: RAG scoring — LLM reads evidence and decides leadership
    leader_jsonl = root / "leader_scores.jsonl"
    leader_summary = root / "leader_scores_summary.json"
    rag_scoring_result = run_rag_scoring(
        input_jsonl=str(merged_desc),
        output_jsonl=str(leader_jsonl),
        output_summary=str(leader_summary),
        top_k=score_top_k,
        model=rag_ollama_model,
        ollama_url=rag_ollama_url,
        timeout_sec=rag_ollama_timeout_sec,
    )

    # Stage 5: memory indexation
    memory_dir = root / "memory"
    run_index(input_jsonl=str(merged_desc), store_dir=str(memory_dir), backend=memory_backend)

    # Stage 6: RAG Q&A (uses LLM-scored candidates as context)
    rag_answer_path = None
    if question:
        simple_store = memory_dir / "simple_store.json"
        if simple_store.exists():
            answer = answer_query(
                question=question,
                scores_jsonl=str(leader_jsonl),
                simple_store_json=str(simple_store),
                top_leaders=top_leaders,
                top_evidence=top_evidence,
                generator_backend=rag_generator_backend,
                ollama_model=rag_ollama_model,
                ollama_url=rag_ollama_url,
                ollama_timeout_sec=rag_ollama_timeout_sec,
            )
            rag_answer_path = root / "rag_answer.json"
            with rag_answer_path.open("w", encoding="utf-8") as f:
                json.dump(answer, f, ensure_ascii=False, indent=2)
        else:
            print("[WARN] Question provided but simple store not available.")

    manifest = {
        "run_name": run_name,
        "videos_count": len(video_paths),
        "videos": [str(p) for p in video_paths],
        "camera_runs": camera_runs,
        "merged_descriptions": str(merged_desc),
        "merged_rows": merged_count,
        "heuristic_scores_jsonl": str(heuristic_jsonl),
        "leader_scores_jsonl": str(leader_jsonl),
        "leader_scoring_backend": rag_scoring_result.get("backend_used", "unknown"),
        "memory_dir": str(memory_dir),
        "rag_answer_json": str(rag_answer_path) if rag_answer_path else None,
    }

    manifest_path = root / "run_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print("Multi-camera pipeline completed")
    print(f"Run root: {root}")
    print(f"Manifest: {manifest_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full multi-camera protest pipeline end-to-end")

    parser.add_argument("--videos", nargs="*", default=[], help="One or more video file paths")
    parser.add_argument("--videos-dir", type=str, default=None, help="Directory containing camera videos")

    parser.add_argument("--run-name", type=str, default="multicam_run")
    parser.add_argument("--output-root", type=str, default="outputs/multicam")

    parser.add_argument("--yolo-model", type=str, default="yolov8n.pt")
    parser.add_argument("--osnet-ckpt", type=str, default="checkpoints/osnet_market1501_gpu_full.pt")
    parser.add_argument("--sim-threshold", type=float, default=0.55)
    parser.add_argument("--max-age", type=int, default=60)
    parser.add_argument("--det-conf", type=float, default=0.25)
    parser.add_argument("--det-iou", type=float, default=0.45)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--descriptor-backend", choices=["auto", "llava", "mock"], default="auto")
    parser.add_argument("--llava-model-id", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--strict-llava", action="store_true")
    parser.add_argument("--sample-every", type=int, default=5)
    parser.add_argument("--max-events", type=int, default=None)

    parser.add_argument("--score-top-k", type=int, default=5)
    parser.add_argument("--memory-backend", choices=["auto", "chroma", "simple"], default="simple")

    parser.add_argument("--question", type=str, default=None)
    parser.add_argument("--top-leaders", type=int, default=3)
    parser.add_argument("--top-evidence", type=int, default=5)
    parser.add_argument("--rag-generator-backend", choices=["template", "ollama"], default="template")
    parser.add_argument("--rag-ollama-model", type=str, default="llama3.1:8b")
    parser.add_argument("--rag-ollama-url", type=str, default="http://127.0.0.1:11434")
    parser.add_argument("--rag-ollama-timeout-sec", type=float, default=60.0)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_multicam_pipeline(
        videos=args.videos,
        videos_dir=args.videos_dir,
        run_name=args.run_name,
        output_root=args.output_root,
        yolo_model=args.yolo_model,
        osnet_ckpt=args.osnet_ckpt,
        sim_threshold=args.sim_threshold,
        max_age=args.max_age,
        det_conf=args.det_conf,
        det_iou=args.det_iou,
        device=args.device,
        descriptor_backend=args.descriptor_backend,
        llava_model_id=args.llava_model_id,
        strict_llava=args.strict_llava,
        sample_every=args.sample_every,
        max_events=args.max_events,
        score_top_k=args.score_top_k,
        memory_backend=args.memory_backend,
        question=args.question,
        top_leaders=args.top_leaders,
        top_evidence=args.top_evidence,
        rag_generator_backend=args.rag_generator_backend,
        rag_ollama_model=args.rag_ollama_model,
        rag_ollama_url=args.rag_ollama_url,
        rag_ollama_timeout_sec=args.rag_ollama_timeout_sec,
    )


if __name__ == "__main__":
    main()
