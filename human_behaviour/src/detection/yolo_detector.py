from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def run_detection(
    source: str,
    model_path: str = "yolov8n.pt",
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 640,
    classes: list[int] | None = None,
    device: str | int | None = None,
    save_dir: str = "outputs/yolo_detect",
) -> None:
    model = YOLO(model_path)
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
    print("YOLO detection completed")
    print(f"Results saved to: {output_root}")
    if results:
        print(f"Processed items: {len(results)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO detection on image/video/folder")
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Input source: image path, video path, folder path, webcam index (0), or URL",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="YOLO model weights path (.pt). Example: yolov8n.pt",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=None,
        help="Optional class filter. Example: --classes 0 for person only",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for inference. Examples: cpu, 0, 0,1",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="outputs/yolo_detect",
        help="Directory where annotated predictions are saved",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_detection(
        source=args.source,
        model_path=args.model,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        classes=args.classes,
        device=args.device,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()
