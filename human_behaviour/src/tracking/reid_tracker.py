from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Set, Tuple

import cv2
import numpy as np
import torch
from torchvision import transforms
from ultralytics import YOLO
from torchreid.reid.models import build_model

# human_behaviour/ (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_under_project(path_str: str) -> Path:
    """Resolve a relative model path against CWD, then human_behaviour/."""
    p = Path(path_str).expanduser()
    if p.is_file():
        return p.resolve()
    alt = PROJECT_ROOT / path_str
    if alt.is_file():
        return alt.resolve()
    return p


@dataclass
class TrackState:
    embedding: np.ndarray
    last_seen_frame: int


class ReIDTracker:
    def __init__(
        self,
        yolo_model_path: str,
        osnet_ckpt_path: str,
        sim_threshold: float = 0.55,
        max_age: int = 60,
        det_conf: float = 0.25,
        det_iou: float = 0.45,
        device: str = "cuda",
    ) -> None:
        self.device = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
        self.sim_threshold = sim_threshold
        self.max_age = max_age
        self.det_conf = det_conf
        self.det_iou = det_iou
        self.next_track_id = 1
        self.tracks: Dict[int, TrackState] = {}

        yolo_path = resolve_under_project(yolo_model_path)
        self.yolo = YOLO(str(yolo_path) if yolo_path.is_file() else yolo_model_path)
        self.osnet, self.embed_transform = self._load_osnet(osnet_ckpt_path)

    def _build_embed_transform(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _load_osnet(self, ckpt_path: str) -> Tuple[torch.nn.Module, transforms.Compose]:
        tfm = self._build_embed_transform()
        env_ckpt = os.getenv("HB_OSNET_CKPT")
        if env_ckpt:
            env_p = Path(env_ckpt).expanduser()
            if env_p.is_file():
                resolved = env_p.resolve()
            else:
                resolved = resolve_under_project(ckpt_path)
        else:
            resolved = resolve_under_project(ckpt_path)

        if resolved.is_file():
            try:
                ckpt = torch.load(resolved, map_location="cpu", weights_only=False)
            except TypeError:
                ckpt = torch.load(resolved, map_location="cpu")
            num_classes = int(ckpt.get("num_classes", 751))

            model = build_model(
                name="osnet_x1_0",
                num_classes=num_classes,
                loss="softmax",
                pretrained=False,
                use_gpu=(self.device.type == "cuda"),
            ).to(self.device)
            model.load_state_dict(ckpt["state_dict"], strict=True)
            model.eval()
            print(f"[OSNet] Loaded checkpoint: {resolved}")
            return model, tfm

        print(
            f"[OSNet] No file at {ckpt_path!r} (tried {resolved}). "
            "Using torchreid ImageNet-pretrained OSNet (downloaded once to ~/.cache/torch/checkpoints)."
        )
        model = build_model(
            name="osnet_x1_0",
            num_classes=1000,
            loss="softmax",
            pretrained=True,
            use_gpu=(self.device.type == "cuda"),
        ).to(self.device)
        model.eval()
        return model, tfm

    def _extract_embedding(self, crop_bgr: np.ndarray) -> np.ndarray:
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.embed_transform(crop_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.osnet(tensor)
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb.squeeze(0).detach().cpu().numpy().astype(np.float32)

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
        return float(np.dot(a, b) / denom)

    def _prune_old_tracks(self, frame_idx: int) -> None:
        to_delete = [tid for tid, st in self.tracks.items() if (frame_idx - st.last_seen_frame) > self.max_age]
        for tid in to_delete:
            del self.tracks[tid]

    def _match_or_create(self, emb: np.ndarray, frame_idx: int, blocked_ids: Set[int]) -> Tuple[int, float, bool]:
        """Returns (track_id, best_similarity_to_matched_or_pruned_pool, is_new_track)."""
        best_id = -1
        best_sim = -1.0

        for track_id, state in self.tracks.items():
            if track_id in blocked_ids:
                continue
            sim = self._cosine_similarity(emb, state.embedding)
            if sim > best_sim:
                best_sim = sim
                best_id = track_id

        if best_id != -1 and best_sim >= self.sim_threshold:
            old = self.tracks[best_id].embedding
            self.tracks[best_id].embedding = 0.8 * old + 0.2 * emb
            self.tracks[best_id].last_seen_frame = frame_idx
            return best_id, best_sim, False

        new_id = self.next_track_id
        self.next_track_id += 1
        self.tracks[new_id] = TrackState(embedding=emb, last_seen_frame=frame_idx)
        return new_id, best_sim, True

    def process_frame(self, frame: np.ndarray, frame_idx: int) -> Tuple[np.ndarray, list[dict], list[dict]]:
        """Returns (annotated_frame, track_events_for_jsonl, raw_yolo_detection_events).

        One video frame = one YOLO pass on the full frame. Each person box gets one OSNet embedding
        and is matched to an existing track or assigned a new track ID (no frame skipping here).
        """
        self._prune_old_tracks(frame_idx)

        results = self.yolo.predict(
            source=frame,
            classes=[0],
            conf=self.det_conf,
            iou=self.det_iou,
            device=(0 if self.device.type == "cuda" else "cpu"),
            verbose=False,
        )

        det_events: list[dict] = []
        yolo_events: list[dict] = []
        if not results:
            return frame, det_events, yolo_events

        boxes = results[0].boxes
        if boxes is None:
            return frame, det_events, yolo_events

        used_track_ids: Set[int] = set()

        for det_ix, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0].item())

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1] - 1, x2)
            y2 = min(frame.shape[0] - 1, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_h, crop_w = crop.shape[0], crop.shape[1]
            yolo_events.append({
                "frame_idx": frame_idx,
                "det_index": det_ix,
                "yolo_class": 0,
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "crop_size": [crop_w, crop_h],
            })

            emb = self._extract_embedding(crop)
            emb_norm = float(np.linalg.norm(emb))
            track_id, sim, is_new = self._match_or_create(emb, frame_idx, blocked_ids=used_track_ids)
            used_track_ids.add(track_id)

            label = f"ID {track_id} | conf {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(frame, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2)

            det_events.append(
                {
                    "frame_idx": frame_idx,
                    "det_index": det_ix,
                    "track_id": track_id,
                    "bbox": [x1, y1, x2, y2],
                    "confidence": conf,
                    "match_similarity": sim,
                    "is_new_track": is_new,
                    "embedding_dim": int(emb.shape[0]),
                    "embedding_l2_norm": round(emb_norm, 6),
                    "embedding_preview": [round(float(x), 6) for x in emb[:8].tolist()],
                }
            )

        return frame, det_events, yolo_events


def run_tracking(
    source: str,
    yolo_model: str,
    osnet_ckpt: str,
    output_dir: str,
    sim_threshold: float,
    max_age: int,
    det_conf: float,
    det_iou: float,
    device: str,
) -> None:
    source_path = Path(source)
    out_root = Path(output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    crops_dir = out_root / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    tracker = ReIDTracker(
        yolo_model_path=yolo_model,
        osnet_ckpt_path=osnet_ckpt,
        sim_threshold=sim_threshold,
        max_age=max_age,
        det_conf=det_conf,
        det_iou=det_iou,
        device=device,
    )

    log_path = out_root / "tracks.jsonl"
    yolo_log = out_root / "yolo_detections.jsonl"

    if source_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        frame = cv2.imread(str(source_path))
        if frame is None:
            raise RuntimeError(f"Could not read image: {source_path}")

        orig = frame.copy()
        vis, events, yolo_raw = tracker.process_frame(frame, frame_idx=0)
        out_img = out_root / "tracked_image.jpg"
        cv2.imwrite(str(out_img), vis)

        # Save detected person crops and enrich logs with crop path.
        for ev in events:
            x1, y1, x2, y2 = ev["bbox"]
            crop = orig[y1:y2, x1:x2]
            if crop.size == 0:
                ev["crop_path"] = None
                continue
            crop_name = f"frame_{ev['frame_idx']:06d}_track_{ev['track_id']:04d}_det_{ev['det_index']:03d}.jpg"
            crop_path = crops_dir / crop_name
            cv2.imwrite(str(crop_path), crop)
            ev["crop_path"] = str(crop_path)

        with log_path.open("w", encoding="utf-8") as f:
            for ev in events:
                f.write(json.dumps(ev) + "\n")
        with yolo_log.open("w", encoding="utf-8") as f:
            for ev in yolo_raw:
                f.write(json.dumps(ev) + "\n")

        summary = {
            "source": str(source_path),
            "mode": "image",
            "frames": 1,
            "yolo_person_detections": len(yolo_raw),
            "track_events_written": len(events),
            "unique_track_ids": sorted({e["track_id"] for e in events}),
            "crops_dir": str(crops_dir),
        }
        with (out_root / "tracking_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"Tracked image saved to: {out_img}")
        print(f"Track log: {log_path}")
        print(f"YOLO raw log: {yolo_log}")
        return

    cap_source = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    out_video_path = out_root / "tracked_video.mp4"
    writer = cv2.VideoWriter(
        str(out_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    frame_idx = 0
    total_yolo = 0
    total_track_events = 0
    total_crops_saved = 0

    with log_path.open("w", encoding="utf-8") as f, yolo_log.open("w", encoding="utf-8") as yf:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            orig = frame.copy()
            vis, events, yolo_raw = tracker.process_frame(frame, frame_idx)
            writer.write(vis)

            for ev in events:
                x1, y1, x2, y2 = ev["bbox"]
                crop = orig[y1:y2, x1:x2]
                if crop.size == 0:
                    ev["crop_path"] = None
                    continue
                crop_name = f"frame_{ev['frame_idx']:06d}_track_{ev['track_id']:04d}_det_{ev['det_index']:03d}.jpg"
                crop_path = crops_dir / crop_name
                cv2.imwrite(str(crop_path), crop)
                ev["crop_path"] = str(crop_path)
                total_crops_saved += 1

            for ev in events:
                f.write(json.dumps(ev) + "\n")
            for ev in yolo_raw:
                yf.write(json.dumps(ev) + "\n")

            total_yolo += len(yolo_raw)
            total_track_events += len(events)
            frame_idx += 1

    cap.release()
    writer.release()

    summary = {
        "source": str(source_path),
        "mode": "video",
        "video_size": [width, height],
        "fps": fps,
        "frames_processed": frame_idx,
        "yolo_person_detections_total": total_yolo,
        "track_events_total": total_track_events,
        "crops_saved_total": total_crops_saved,
        "crops_dir": str(crops_dir),
        "note": "Per frame: YOLO class 0 (person), then OSNet embedding per valid box; no frame skipping in tracker.",
    }
    with (out_root / "tracking_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 60)
    print("YOLO + OSNet ReID tracking completed")
    print(f"Tracked video: {out_video_path}")
    print(f"Track log (YOLO + ReID merged): {log_path}")
    print(f"YOLO-only log: {yolo_log}")
    print(f"Summary: {out_root / 'tracking_summary.json'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track people with YOLO detections + OSNet ReID embeddings")
    parser.add_argument("--source", type=str, required=True, help="Image/video path or camera index")
    parser.add_argument("--yolo-model", type=str, default="yolov8n.pt", help="YOLO weights path")
    parser.add_argument(
        "--osnet-ckpt",
        type=str,
        default="checkpoints/osnet_market1501_gpu_full.pt",
        help="OSNet checkpoint path",
    )
    parser.add_argument("--output-dir", type=str, default="outputs/reid_tracking")
    parser.add_argument("--sim-threshold", type=float, default=0.55)
    parser.add_argument("--max-age", type=int, default=60, help="Frames to keep unseen tracks")
    parser.add_argument("--det-conf", type=float, default=0.25, help="YOLO person detection confidence threshold")
    parser.add_argument("--det-iou", type=float, default=0.45, help="YOLO NMS IoU threshold")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_tracking(
        source=args.source,
        yolo_model=args.yolo_model,
        osnet_ckpt=args.osnet_ckpt,
        output_dir=args.output_dir,
        sim_threshold=args.sim_threshold,
        max_age=args.max_age,
        det_conf=args.det_conf,
        det_iou=args.det_iou,
        device=args.device,
    )


if __name__ == "__main__":
    main()
