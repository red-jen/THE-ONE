from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.market1501_dataset import Market1501Dataset
from torchreid.reid.models import build_model


def build_dataloader(dataset_root: str, batch_size: int, num_workers: int) -> tuple[DataLoader, int]:
    train_tfms = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = Market1501Dataset(root=dataset_root, split="train", transform=train_tfms)
    num_classes = len(train_ds.id_to_label)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    return train_loader, num_classes


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int | None,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    for batch_idx, batch in enumerate(tqdm(loader, desc="Train", leave=False)):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        preds = torch.argmax(logits, dim=1)
        running_correct += (preds == labels).sum().item()
        running_total += labels.size(0)

    avg_loss = running_loss / max(1, running_total)
    avg_acc = running_correct / max(1, running_total)
    return avg_loss, avg_acc


def main() -> None:
    parser = argparse.ArgumentParser(description="Train OSNet on Market-1501 train split")
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--max-batches", type=int, default=50)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--save-path", type=str, default="checkpoints/osnet_market1501_sanity.pt")
    parser.add_argument("--mlflow", action="store_true", help="Enable MLflow logging")
    parser.add_argument("--mlflow-tracking-uri", type=str, default=None, help="Optional MLflow tracking URI")
    parser.add_argument("--mlflow-experiment", type=str, default="human_behaviour_osnet")
    parser.add_argument("--mlflow-run-name", type=str, default=None)
    args = parser.parse_args()

    mlflow = None
    if args.mlflow:
        try:
            mlflow = importlib.import_module("mlflow")
            if args.mlflow_tracking_uri:
                mlflow.set_tracking_uri(args.mlflow_tracking_uri)
            mlflow.set_experiment(args.mlflow_experiment)
            mlflow.start_run(run_name=args.mlflow_run_name)
            mlflow.log_param("dataset_root", args.dataset_root)
            mlflow.log_param("epochs", args.epochs)
            mlflow.log_param("batch_size", args.batch_size)
            mlflow.log_param("num_workers", args.num_workers)
            mlflow.log_param("lr", args.lr)
            mlflow.log_param("weight_decay", args.weight_decay)
            mlflow.log_param("max_batches", args.max_batches)
            mlflow.log_param("pretrained", bool(args.pretrained))
            mlflow.log_param("save_path", args.save_path)
        except Exception as exc:
            print(f"[WARN] MLflow requested but unavailable: {exc}")
            mlflow = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    loader, num_classes = build_dataloader(args.dataset_root, args.batch_size, args.num_workers)
    print(f"Number of train identities (classes): {num_classes}")

    model = build_model(
        name="osnet_x1_0",
        num_classes=num_classes,
        loss="softmax",
        pretrained=args.pretrained,
        use_gpu=torch.cuda.is_available(),
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(1, args.epochs + 1):
        loss, acc = train_one_epoch(model, loader, optimizer, criterion, device, args.max_batches)
        print(f"Epoch {epoch}/{args.epochs} | loss={loss:.4f} | acc={acc:.4f}")
        if mlflow is not None:
            mlflow.log_metric("train_loss", float(loss), step=epoch)
            mlflow.log_metric("train_acc", float(acc), step=epoch)

    save_path = PROJECT_ROOT / args.save_path
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": "osnet_x1_0",
            "num_classes": num_classes,
            "state_dict": model.state_dict(),
        },
        save_path,
    )
    print(f"Checkpoint saved to: {save_path}")

    if mlflow is not None:
        mlflow.log_param("num_classes", num_classes)
        mlflow.log_artifact(str(save_path.resolve()))
        mlflow.end_run()


if __name__ == "__main__":
    main()
