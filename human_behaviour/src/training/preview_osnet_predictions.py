from __future__ import annotations

import argparse
import random
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.market1501_dataset import Market1501Dataset
from torchreid.reid.models import build_model


def build_eval_dataset(dataset_root: str, split: str) -> Market1501Dataset:
    eval_tfms = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return Market1501Dataset(root=dataset_root, split=split, transform=eval_tfms)


def extract_features(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> torch.Tensor:
    model.eval()
    features = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            feats = model(images)
            features.append(feats.cpu())
    return torch.cat(features, dim=0)


def draw_panel(query_path: str, gallery_paths: list[str], hit_flags: list[bool], out_path: Path) -> None:
    query_img = Image.open(query_path).convert("RGB")
    panel_items = [query_img] + [Image.open(p).convert("RGB") for p in gallery_paths]

    w, h = 128, 256
    margin = 10
    title_h = 24
    total_w = margin + len(panel_items) * (w + margin)
    total_h = title_h + h + margin
    canvas = Image.new("RGB", (total_w, total_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    for idx, img in enumerate(panel_items):
        x = margin + idx * (w + margin)
        y = title_h
        canvas.paste(img.resize((w, h)), (x, y))
        if idx == 0:
            draw.text((x, 4), "QUERY", fill=(0, 0, 255))
        else:
            color = (0, 170, 0) if hit_flags[idx - 1] else (220, 0, 0)
            draw.rectangle([x, y, x + w - 1, y + h - 1], outline=color, width=3)
            draw.text((x, 4), f"R{idx}", fill=color)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview OSNet query-gallery predictions")
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/osnet_market1501_sanity.pt")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--gallery-max", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs/osnet_preview")
    args = parser.parse_args()

    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint_path = PROJECT_ROOT / args.checkpoint
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    query_ds = build_eval_dataset(args.dataset_root, "query")
    gallery_ds_full = build_eval_dataset(args.dataset_root, "gallery")

    if args.gallery_max is not None and args.gallery_max > 0 and len(gallery_ds_full) > args.gallery_max:
        indices = list(range(len(gallery_ds_full)))
        random.shuffle(indices)
        indices = indices[: args.gallery_max]
        gallery_ds = Subset(gallery_ds_full, indices)
        gallery_info = [gallery_ds_full.images[i] for i in indices]
    else:
        gallery_ds = gallery_ds_full
        gallery_info = gallery_ds_full.images

    query_loader = DataLoader(query_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    gallery_loader = DataLoader(gallery_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    num_classes = len(query_ds.id_to_label)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if "num_classes" in ckpt:
        num_classes = int(ckpt["num_classes"])

    model = build_model(
        name="osnet_x1_0",
        num_classes=num_classes,
        loss="softmax",
        pretrained=False,
        use_gpu=torch.cuda.is_available(),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    print("Extracting query features...")
    q_feats = extract_features(model, query_loader, device)
    print("Extracting gallery features...")
    g_feats = extract_features(model, gallery_loader, device)

    q_feats = torch.nn.functional.normalize(q_feats, p=2, dim=1)
    g_feats = torch.nn.functional.normalize(g_feats, p=2, dim=1)

    num_queries = len(query_ds)
    sample_count = min(args.num_samples, num_queries)
    sampled_q = random.sample(range(num_queries), sample_count)

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    report_lines = []

    for idx, q_idx in enumerate(sampled_q, start=1):
        q_info = query_ds.images[q_idx]
        sims = torch.matmul(q_feats[q_idx : q_idx + 1], g_feats.T).squeeze(0)
        order = torch.argsort(sims, descending=True).tolist()

        valid_order = []
        for g_idx in order:
            g_info = gallery_info[g_idx]
            if g_info.person_id == q_info.person_id and g_info.camera_id == q_info.camera_id:
                continue
            valid_order.append(g_idx)
            if len(valid_order) >= args.topk:
                break

        top_gallery = [gallery_info[g] for g in valid_order]
        hits = [g.person_id == q_info.person_id for g in top_gallery]

        panel_path = output_dir / f"sample_{idx:02d}.jpg"
        draw_panel(q_info.path, [g.path for g in top_gallery], hits, panel_path)

        report_lines.append(
            f"Sample {idx}: query pid={q_info.person_id}, cam={q_info.camera_id}, "
            f"top-{args.topk} hits={sum(hits)}/{args.topk}, panel={panel_path.name}"
        )

    report_path = output_dir / "report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print("\n" + "=" * 60)
    print("Prediction preview generated")
    print(f"Output folder: {output_dir}")
    print(f"Report: {report_path}")
    for line in report_lines:
        print(line)


if __name__ == "__main__":
    main()
