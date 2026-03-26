from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def run_off_the_shelf_detection(
    source: str,
    model_path: str = "yolov8n.pt",
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 640,
    person_only: bool = True,
    device: str | int | None = "0",
    save_dir: str = "outputs/yolo_off_the_shelf",
) -> None:
    """Run detection with pretrained YOLO weights (no fine-tuning)."""
    model = YOLO(model_path)

    classes = [0] if person_only else None  # COCO class 0 = person
    project_dir = Path(save_dir).resolve()

    results = model.predict(
        source=source,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        classes=classes,
        device=device,
        save=True,
        project=str(project_dir),
        name="predictions",
        exist_ok=True,
        verbose=True,
    )

    output_root = project_dir / "predictions"
    print("=" * 60)
    print("Off-the-shelf YOLO detection completed")
    print(f"Model: {model_path}")
    print(f"Source: {source}")
    print(f"Person-only mode: {person_only}")
    print(f"Results saved to: {output_root}")
    if results:
        print(f"Processed items: {len(results)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Off-the-shelf YOLO inference (pretrained, no training required)"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Input source: image path, video path, folder, webcam index, or URL",
    )
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Pretrained YOLO weights")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument(
        "--all-classes",
        action="store_true",
        help="Detect all classes (default is person-only)",
    )
    parser.add_argument("--device", type=str, default="0", help="Device: cpu, 0, 0,1")
    parser.add_argument(
        "--save-dir",
        type=str,
        default="outputs/yolo_off_the_shelf",
        help="Directory where annotated predictions are saved",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_off_the_shelf_detection(
        source=args.source,
        model_path=args.model,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        person_only=not args.all_classes,
        device=args.device,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()
