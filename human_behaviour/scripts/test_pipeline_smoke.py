from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_cmd(command: list[str], title: str) -> None:
    print("=" * 80)
    print(f"[RUN] {title}")
    print(" ".join(command))
    proc = subprocess.run(command, cwd=str(PROJECT_ROOT), check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({title}) with code {proc.returncode}")
    print(f"[OK] {title}")


def ensure_synthetic_video(sample_image: Path, out_video: Path, frames: int = 30, fps: float = 10.0) -> None:
    if out_video.exists():
        return

    img = cv2.imread(str(sample_image))
    if img is None:
        raise RuntimeError(f"Could not read sample image: {sample_image}")

    out_video.parent.mkdir(parents=True, exist_ok=True)
    h, w = img.shape[:2]
    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for _ in range(frames):
        writer.write(img)
    writer.release()


def assert_file(path: Path, label: str) -> None:
    if not path.exists():
        raise RuntimeError(f"Missing expected output for {label}: {path}")
    print(f"[PASS] {label}: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Automated smoke test for the protest pipeline")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument(
        "--sample-image",
        type=str,
        default="runs/detect/outputs/yolo_detect/predictions/0001_c1s1_001051_00.jpg",
    )
    parser.add_argument("--yolo-model", type=str, default="yolov8n.pt")
    parser.add_argument("--osnet-ckpt", type=str, default="checkpoints/osnet_market1501_gpu_full.pt")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    sample_image = PROJECT_ROOT / args.sample_image
    synthetic_video = PROJECT_ROOT / "outputs/reid_tracking_smoke/synthetic_input.mp4"
    ensure_synthetic_video(sample_image=sample_image, out_video=synthetic_video)

    yolo_out = PROJECT_ROOT / "outputs/yolo_off_the_shelf_smoke/predictions"
    run_cmd(
        [
            args.python,
            "src/detection/yolo_off_the_shelf.py",
            "--source",
            str(sample_image),
            "--model",
            args.yolo_model,
            "--conf",
            "0.25",
            "--iou",
            "0.45",
            "--device",
            "0" if args.device == "cuda" else "cpu",
            "--save-dir",
            "outputs/yolo_off_the_shelf_smoke",
        ],
        "YOLO detection smoke",
    )
    assert_file(yolo_out / sample_image.name, "YOLO annotated image")

    track_dir = PROJECT_ROOT / "outputs/reid_tracking_smoke/test_run_auto"
    run_cmd(
        [
            args.python,
            "src/tracking/reid_tracker.py",
            "--source",
            str(synthetic_video),
            "--yolo-model",
            args.yolo_model,
            "--osnet-ckpt",
            args.osnet_ckpt,
            "--output-dir",
            str(track_dir),
            "--sim-threshold",
            "0.55",
            "--max-age",
            "60",
            "--det-conf",
            "0.01",
            "--det-iou",
            "0.45",
            "--device",
            args.device,
        ],
        "ReID tracker smoke",
    )
    assert_file(track_dir / "tracked_video.mp4", "Tracked video")
    assert_file(track_dir / "tracks.jsonl", "Track log")

    desc_jsonl = PROJECT_ROOT / "outputs/descriptions/test_run_auto_descriptions.jsonl"
    desc_summary = PROJECT_ROOT / "outputs/descriptions/test_run_auto_summary.json"
    run_cmd(
        [
            args.python,
            "src/description/llava_descriptor.py",
            "--source",
            str(synthetic_video),
            "--tracks-jsonl",
            str(track_dir / "tracks.jsonl"),
            "--output-jsonl",
            str(desc_jsonl),
            "--output-summary",
            str(desc_summary),
            "--camera-id",
            "cam_test_auto",
            "--backend",
            "mock",
            "--device",
            args.device,
            "--sample-every",
            "5",
        ],
        "Description smoke",
    )
    assert_file(desc_jsonl, "Description JSONL")
    assert_file(desc_summary, "Description summary")

    score_jsonl = PROJECT_ROOT / "outputs/scores/test_run_auto_scores.jsonl"
    score_summary = PROJECT_ROOT / "outputs/scores/test_run_auto_scores_summary.json"
    run_cmd(
        [
            args.python,
            "src/scoring/leader_scorer.py",
            "--input-jsonl",
            str(desc_jsonl),
            "--output-jsonl",
            str(score_jsonl),
            "--output-summary",
            str(score_summary),
            "--top-k",
            "3",
        ],
        "Leader scoring smoke",
    )
    assert_file(score_jsonl, "Scores JSONL")
    assert_file(score_summary, "Scores summary")

    memory_dir = PROJECT_ROOT / "outputs/memory/test_run_auto"
    run_cmd(
        [
            args.python,
            "src/memory/chroma_store.py",
            "index",
            "--input-jsonl",
            str(desc_jsonl),
            "--store-dir",
            str(memory_dir),
            "--backend",
            "simple",
        ],
        "Memory index smoke",
    )
    assert_file(memory_dir / "index_summary.json", "Memory index summary")

    rag_json = PROJECT_ROOT / "outputs/rag/test_run_auto_answer.json"
    run_cmd(
        [
            args.python,
            "src/interface/chatbot.py",
            "--question",
            "who is most likely leading and why",
            "--scores-jsonl",
            str(score_jsonl),
            "--simple-store",
            str(memory_dir / "simple_store.json"),
            "--top-leaders",
            "2",
            "--top-evidence",
            "3",
            "--output-json",
            str(rag_json),
        ],
        "RAG query smoke",
    )
    assert_file(rag_json, "RAG answer JSON")

    print("=" * 80)
    print("ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
