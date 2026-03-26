from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class TrackEvent:
    frame_idx: int
    track_id: int
    bbox: Tuple[int, int, int, int]
    confidence: float
    match_similarity: float


class DescriptorBackend:
    def describe(self, crop_bgr: np.ndarray, prompt: str) -> str:
        raise NotImplementedError


class MockDescriptorBackend(DescriptorBackend):
    def describe(self, crop_bgr: np.ndarray, prompt: str) -> str:
        height, width = crop_bgr.shape[:2]
        aspect = width / max(height, 1)
        posture = "upright" if aspect < 0.8 else "side-facing"
        return (
            f"A person appears {posture} in a crowded scene. "
            f"Upper-body crop quality is moderate. "
            f"No clear protest object is confidently visible."
        )


class LlavaDescriptorBackend(DescriptorBackend):
    def __init__(self, model_id: str, device: str, max_new_tokens: int) -> None:
        self.max_new_tokens = max_new_tokens
        self.device = device

        try:
            import torch
        except Exception as exc:  # pragma: no cover - runtime dependency path
            raise RuntimeError(
                "LLaVA backend requires transformers + torch. "
                "Install transformers and ensure compatible model weights are available."
            ) from exc

        try:
            transformers_mod = importlib.import_module("transformers")
            AutoProcessor = getattr(transformers_mod, "AutoProcessor")
            LlavaForConditionalGeneration = getattr(transformers_mod, "LlavaForConditionalGeneration")
        except Exception as exc:  # pragma: no cover - runtime dependency path
            raise RuntimeError(
                "Transformers with LlavaForConditionalGeneration is required for LLaVA backend."
            ) from exc

        self.torch = torch
        dtype = torch.float16 if (device == "cuda" and torch.cuda.is_available()) else torch.float32

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = LlavaForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype)
        if device == "cuda" and torch.cuda.is_available():
            self.model = self.model.to("cuda")
        self.model.eval()

    def _build_prompt(self, instruction: str) -> str:
        return f"USER: <image>\n{instruction}\nASSISTANT:"

    def describe(self, crop_bgr: np.ndarray, prompt: str) -> str:
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

        pil = None
        try:
            from PIL import Image

            pil = Image.fromarray(crop_rgb)
        except Exception as exc:  # pragma: no cover - runtime dependency path
            raise RuntimeError("Pillow is required for LLaVA image preprocessing.") from exc

        text_prompt = self._build_prompt(prompt)
        inputs = self.processor(images=pil, text=text_prompt, return_tensors="pt")

        if self.device == "cuda" and self.torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with self.torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        decoded = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        if "ASSISTANT:" in decoded:
            return decoded.split("ASSISTANT:", 1)[1].strip()
        return decoded


def build_backend(
    backend: str,
    model_id: str,
    device: str,
    max_new_tokens: int,
    strict_llava: bool,
) -> DescriptorBackend:
    if backend == "mock":
        return MockDescriptorBackend()

    if backend in {"llava", "auto"}:
        try:
            return LlavaDescriptorBackend(model_id=model_id, device=device, max_new_tokens=max_new_tokens)
        except Exception as exc:
            if backend == "llava" or strict_llava:
                raise
            print(f"[WARN] LLaVA backend unavailable, falling back to mock. Reason: {exc}")
            return MockDescriptorBackend()

    raise ValueError(f"Unsupported backend: {backend}")


def load_track_events(
    tracks_jsonl: str,
    sample_every: int = 1,
    max_events: Optional[int] = None,
) -> List[TrackEvent]:
    events: List[TrackEvent] = []
    with Path(tracks_jsonl).open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            frame_idx = int(row["frame_idx"])
            if sample_every > 1 and (frame_idx % sample_every != 0):
                continue

            bbox = row.get("bbox", [0, 0, 0, 0])
            events.append(
                TrackEvent(
                    frame_idx=frame_idx,
                    track_id=int(row["track_id"]),
                    bbox=(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
                    confidence=float(row.get("confidence", -1.0)),
                    match_similarity=float(row.get("match_similarity", -1.0)),
                )
            )
            if max_events is not None and len(events) >= max_events:
                break

    events.sort(key=lambda e: (e.frame_idx, e.track_id))
    return events


def group_events_by_frame(events: Iterable[TrackEvent]) -> Dict[int, List[TrackEvent]]:
    grouped: Dict[int, List[TrackEvent]] = {}
    for ev in events:
        grouped.setdefault(ev.frame_idx, []).append(ev)
    return grouped


def clamp_bbox(bbox: Tuple[int, int, int, int], width: int, height: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))
    return x1, y1, x2, y2


def extract_signals(description: str) -> Dict[str, object]:
    text = description.lower()

    has_megaphone = "megaphone" in text
    has_banner = ("banner" in text) or ("sign" in text) or ("placard" in text)
    has_flag = "flag" in text
    has_microphone = "microphone" in text or "mic" in text

    gesture_keywords = ["gesture", "point", "raised hand", "addressing", "speaking"]
    gesture_score = sum(1 for k in gesture_keywords if k in text)

    if "front" in text:
        position = "front"
    elif "center" in text or "middle" in text:
        position = "center"
    elif "back" in text or "rear" in text:
        position = "back"
    else:
        position = "unknown"

    return {
        "has_megaphone": has_megaphone,
        "has_banner": has_banner,
        "has_flag": has_flag,
        "has_microphone": has_microphone,
        "gesture_score": gesture_score,
        "position_hint": position,
    }


def run_description_pipeline(
    source: str,
    tracks_jsonl: str,
    output_jsonl: str,
    output_summary: str,
    camera_id: str,
    prompt: str,
    backend_name: str,
    model_id: str,
    device: str,
    max_new_tokens: int,
    sample_every: int,
    max_events: Optional[int],
    strict_llava: bool,
) -> None:
    events = load_track_events(tracks_jsonl=tracks_jsonl, sample_every=sample_every, max_events=max_events)
    if not events:
        raise RuntimeError(f"No events loaded from tracks file: {tracks_jsonl}")

    backend = build_backend(
        backend=backend_name,
        model_id=model_id,
        device=device,
        max_new_tokens=max_new_tokens,
        strict_llava=strict_llava,
    )

    output_jsonl_path = Path(output_jsonl).resolve()
    output_summary_path = Path(output_summary).resolve()
    output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    output_summary_path.parent.mkdir(parents=True, exist_ok=True)

    grouped = group_events_by_frame(events)

    total = 0
    per_person_counts: Dict[int, int] = {}

    source_path = Path(source)
    if source_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        frame = cv2.imread(str(source_path))
        if frame is None:
            raise RuntimeError(f"Could not read image source: {source}")

        frame_events = grouped.get(0, [])
        with output_jsonl_path.open("w", encoding="utf-8") as out_f:
            for ev in frame_events:
                x1, y1, x2, y2 = clamp_bbox(ev.bbox, width=frame.shape[1], height=frame.shape[0])
                if x2 <= x1 or y2 <= y1:
                    continue

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                desc = backend.describe(crop_bgr=crop, prompt=prompt)
                signals = extract_signals(desc)

                row = {
                    "person_id": ev.track_id,
                    "frame_idx": ev.frame_idx,
                    "timestamp_sec": 0.0,
                    "camera_id": camera_id,
                    "source": str(source_path),
                    "bbox": [x1, y1, x2, y2],
                    "detection_confidence": ev.confidence,
                    "match_similarity": ev.match_similarity,
                    "description": desc,
                    "signals": signals,
                }
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")

                total += 1
                per_person_counts[ev.track_id] = per_person_counts.get(ev.track_id, 0) + 1
    else:
        cap_source = int(source) if source.isdigit() else source
        cap = cv2.VideoCapture(cap_source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open source: {source}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25.0

        wanted_frames = set(grouped.keys())
        frame_idx = 0

        with output_jsonl_path.open("w", encoding="utf-8") as out_f:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx not in wanted_frames:
                    frame_idx += 1
                    continue

                frame_events = grouped.get(frame_idx, [])
                for ev in frame_events:
                    x1, y1, x2, y2 = clamp_bbox(ev.bbox, width=frame.shape[1], height=frame.shape[0])
                    if x2 <= x1 or y2 <= y1:
                        continue

                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue

                    desc = backend.describe(crop_bgr=crop, prompt=prompt)
                    signals = extract_signals(desc)

                    row = {
                        "person_id": ev.track_id,
                        "frame_idx": ev.frame_idx,
                        "timestamp_sec": frame_idx / fps,
                        "camera_id": camera_id,
                        "source": str(source),
                        "bbox": [x1, y1, x2, y2],
                        "detection_confidence": ev.confidence,
                        "match_similarity": ev.match_similarity,
                        "description": desc,
                        "signals": signals,
                    }
                    out_f.write(json.dumps(row, ensure_ascii=False) + "\n")

                    total += 1
                    per_person_counts[ev.track_id] = per_person_counts.get(ev.track_id, 0) + 1

                frame_idx += 1

        cap.release()

    summary = {
        "source": str(source),
        "tracks_jsonl": str(Path(tracks_jsonl).resolve()),
        "backend": backend_name,
        "model_id": model_id if backend_name != "mock" else "mock",
        "camera_id": camera_id,
        "events_loaded": len(events),
        "descriptions_written": total,
        "unique_person_ids": sorted(per_person_counts.keys()),
        "descriptions_per_person": {str(k): v for k, v in sorted(per_person_counts.items())},
    }

    with output_summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=" * 60)
    print("LLaVA description pipeline completed")
    print(f"Backend used: {backend_name}")
    print(f"Descriptions JSONL: {output_jsonl_path}")
    print(f"Summary JSON: {output_summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate per-person behavior descriptions from tracker outputs")

    parser.add_argument("--source", type=str, required=True, help="Image/video path or camera index used in tracking")
    parser.add_argument("--tracks-jsonl", type=str, required=True, help="Tracker events JSONL file")

    parser.add_argument(
        "--output-jsonl",
        type=str,
        default="outputs/descriptions/descriptions.jsonl",
        help="Path to description events JSONL",
    )
    parser.add_argument(
        "--output-summary",
        type=str,
        default="outputs/descriptions/summary.json",
        help="Path to summary JSON",
    )

    parser.add_argument("--camera-id", type=str, default="cam_1")
    parser.add_argument(
        "--prompt",
        type=str,
        default=(
            "Describe this person's actions, position in the crowd, visible objects, and body language. "
            "If uncertain, state uncertainty explicitly."
        ),
    )

    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "llava", "mock"])
    parser.add_argument("--model-id", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--strict-llava", action="store_true", help="Fail if LLaVA cannot load")

    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--max-new-tokens", type=int, default=96)

    parser.add_argument("--sample-every", type=int, default=1, help="Only process every Nth frame index")
    parser.add_argument("--max-events", type=int, default=None, help="Optional cap on number of events")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_description_pipeline(
        source=args.source,
        tracks_jsonl=args.tracks_jsonl,
        output_jsonl=args.output_jsonl,
        output_summary=args.output_summary,
        camera_id=args.camera_id,
        prompt=args.prompt,
        backend_name=args.backend,
        model_id=args.model_id,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        sample_every=args.sample_every,
        max_events=args.max_events,
        strict_llava=args.strict_llava,
    )


if __name__ == "__main__":
    main()
