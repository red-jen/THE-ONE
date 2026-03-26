from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def train_yolo(
    data_yaml: str,
    model_path: str = "yolov8n.pt",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    device: str | int | None = "0",
    workers: int = 8,
    project: str = "outputs/yolo_train",
    name: str = "protest_detector",
    patience: int = 30,
    cache: bool = False,
    resume: bool = False,
) -> None:
    data_path = Path(data_yaml)
    if not data_path.exists():
        raise FileNotFoundError(f"YOLO data yaml not found: {data_path}")

    model = YOLO(model_path)

    results = model.train(
        data=str(data_path.resolve()),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        project=str(Path(project).resolve()),
        name=name,
        exist_ok=True,
        patience=patience,
        cache=cache,
        resume=resume,
        pretrained=True,
        verbose=True,
    )

    save_dir = getattr(results, "save_dir", None)
    print("=" * 60)
    print("YOLO training completed")
    if save_dir is not None:
        print(f"Run directory: {save_dir}")
        print(f"Best checkpoint: {Path(save_dir) / 'weights' / 'best.pt'}")
        print(f"Last checkpoint: {Path(save_dir) / 'weights' / 'last.pt'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO on a custom detection dataset")
    parser.add_argument("--data", type=str, required=True, help="Path to YOLO data yaml")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Base model weights")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="0", help="Device: cpu, 0, 0,1")
    parser.add_argument("--workers", type=int, default=8, help="Data loader workers")
    parser.add_argument("--project", type=str, default="outputs/yolo_train", help="Output project folder")
    parser.add_argument("--name", type=str, default="protest_detector", help="Run name")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience")
    parser.add_argument("--cache", action="store_true", help="Cache images in RAM for faster training")
    parser.add_argument("--resume", action="store_true", help="Resume last run if available")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_yolo(
        data_yaml=args.data,
        model_path=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        patience=args.patience,
        cache=args.cache,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
