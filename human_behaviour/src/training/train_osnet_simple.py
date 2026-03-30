from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
import torch.nn as nn
from torchreid.reid.models import build_model
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.simple_market1501 import build_simple_train_loader


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in tqdm(loader, desc="Training", leave=False):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += labels.size(0)

    avg_loss = total_loss / max(1, total_samples)
    avg_acc = total_correct / max(1, total_samples)
    return avg_loss, avg_acc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal OSNet training on Market-1501")
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save-path", type=str, default="checkpoints/osnet_market1501_simple.pt")
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda; default auto")
    parser.add_argument("--pretrained", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    train_loader, num_classes = build_simple_train_loader(
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = build_model(
        name="osnet_x1_0",
        num_classes=num_classes,
        loss="softmax",
        pretrained=args.pretrained,
        use_gpu=(device.type == "cuda"),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"Device: {device}")
    print(f"Classes: {num_classes}")
    print(f"Batches per epoch: {len(train_loader)}")

    for epoch in range(1, args.epochs + 1):
        loss, acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch}/{args.epochs} - loss={loss:.4f} acc={acc:.4f}")

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
    print(f"Saved checkpoint: {save_path}")


if __name__ == "__main__":
    main()
