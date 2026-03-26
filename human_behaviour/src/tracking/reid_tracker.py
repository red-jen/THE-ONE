from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Set, Tuple

import cv2
import numpy as np
import torch
from torchvision import transforms
from ultralytics import YOLO
from torchreid.reid.models import build_model


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

        self.yolo = YOLO(yolo_model_path)
        self.osnet, self.embed_transform = self._load_osnet(osnet_ckpt_path)

    def _load_osnet(self, ckpt_path: str) -> Tuple[torch.nn.Module, transforms.Compose]:
        ckpt = torch.load(ckpt_path, map_location="cpu")
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

        tfm = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
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

    def _match_or_create(self, emb: np.ndarray, frame_idx: int, blocked_ids: Set[int]) -> Tuple[int, float]:
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
            return best_id, best_sim

        new_id = self.next_track_id
        self.next_track_id += 1
        self.tracks[new_id] = TrackState(embedding=emb, last_seen_frame=frame_idx)
        return new_id, best_sim

    def process_frame(self, frame: np.ndarray, frame_idx: int) -> Tuple[np.ndarray, list[dict]]:
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
        if not results:
            return frame, det_events

        boxes = results[0].boxes
        if boxes is None:
            return frame, det_events

        used_track_ids: Set[int] = set()

        for box in boxes:
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

            emb = self._extract_embedding(crop)
            track_id, sim = self._match_or_create(emb, frame_idx, blocked_ids=used_track_ids)
            used_track_ids.add(track_id)

            label = f"ID {track_id} | conf {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(frame, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2)

            det_events.append(
                {
                    "frame_idx": frame_idx,
                    "track_id": track_id,
                    "bbox": [x1, y1, x2, y2],
                    "confidence": conf,
                    "match_similarity": sim,
                }
            )

        return frame, det_events


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

    if source_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        frame = cv2.imread(str(source_path))
        if frame is None:
            raise RuntimeError(f"Could not read image: {source_path}")

        vis, events = tracker.process_frame(frame, frame_idx=0)
        out_img = out_root / "tracked_image.jpg"
        cv2.imwrite(str(out_img), vis)

        with log_path.open("w", encoding="utf-8") as f:
            for ev in events:
                f.write(json.dumps(ev) + "\n")

        print(f"Tracked image saved to: {out_img}")
        print(f"Track log saved to: {log_path}")
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
    with log_path.open("w", encoding="utf-8") as f:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            vis, events = tracker.process_frame(frame, frame_idx)
            writer.write(vis)

            for ev in events:
                f.write(json.dumps(ev) + "\n")

            frame_idx += 1

    cap.release()
    writer.release()

    print("=" * 60)
    print("YOLO + OSNet ReID tracking completed")
    print(f"Tracked video: {out_video_path}")
    print(f"Track log: {log_path}")


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
