from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from data.market1501_dataset import Market1501Dataset


def validate_market1501_root(dataset_root: str) -> None:
    root = Path(dataset_root)
    required = [
        root / "bounding_box_train",
        root / "bounding_box_test",
        root / "query",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Market-1501 folders missing: " + ", ".join(missing)
        )


def build_simple_train_loader(
    dataset_root: str,
    batch_size: int = 16,
    num_workers: int = 0,
) -> tuple[DataLoader, int]:
    validate_market1501_root(dataset_root)

    transform = transforms.Compose(
        [
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    train_dataset = Market1501Dataset(
        root=dataset_root,
        split="train",
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    num_classes = len(train_dataset.id_to_label)
    return train_loader, num_classes
