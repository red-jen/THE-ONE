# Protest Leader Detection — System Architecture (Current State)

## Project Goal
Detect and identify protest leaders from multi-camera footage by combining:
1) person detection, 2) re-identification, 3) behavior description, 4) leader scoring, 5) retrieval/query interface.

---

## Implemented Pipeline (Working)

### Module 1 — Detection + ReID Tracking ✅
- YOLO person detection + OSNet embedding matching with persistent IDs.
- Code:
  - `src/detection/yolo_off_the_shelf.py`
  - `src/tracking/reid_tracker.py`
- Output:
  - annotated video/image
  - `tracks.jsonl` with `{frame_idx, track_id, bbox, confidence, match_similarity}`

### Module 2 — Behavior Description ✅
- Generates per-person descriptions from tracked crops.
- Backends: `mock`, `auto`, `llava` (fallback behavior implemented).
- Code: `src/description/llava_descriptor.py`
- Output: description JSONL + summary JSON.

### Module 3 — Leader Scoring ✅
- Aggregates signals by person and computes interpretable leader score components.
- Code: `src/scoring/leader_scorer.py`
- Output: ranked score JSONL + summary.

### Module 4 — Memory Index + Query ✅
- Index + retrieval over generated descriptions.
- Backends: `simple`, `auto`, `chroma` (if available).
- Code: `src/memory/chroma_store.py`

### Module 5 — Interfaces ✅
- CLI RAG-style query: `src/interface/chatbot.py`
- FastAPI endpoints + upload pipeline trigger: `src/interface/api.py`
- Streamlit drag/drop UI: `src/interface/streamlit_app.py`

### Module 6 — End-to-End Orchestrator ✅
- One-command multi-camera run:
  - track -> describe -> score -> index -> answer
- Code: `src/pipeline/run_multicam_pipeline.py`

### Module 7 — MLOps + Security Baseline ✅
- MLflow logging integrated in training/scoring.
- JWT auth added to API (`/auth/token`, protected endpoints, env-toggle).
- Automated smoke test script added: `scripts/test_pipeline_smoke.py`.

---

## Technology Stack (Real)

| Component | Current technology |
|---|---|
| Detection | Ultralytics YOLO (`yolov8n.pt` baseline) |
| Re-ID | OSNet via `torchreid` |
| Description | LLaVA path + mock fallback |
| Scoring | Custom weighted rules |
| Memory | Simple lexical store + optional Chroma |
| Query | Local RAG-style answer synthesis |
| API | FastAPI |
| UI | Streamlit |
| MLOps | MLflow (Docker endpoint supported) |

---

## Current Maturity by Step

| Step | Status | Notes |
|---|---|---|
| Dataset preparation | ✅ Done | Market-1501 tooling complete |
| OSNet training | ✅ Done | Training + checkpoints + smoke validation |
| YOLO detection | ✅ Baseline done | Fine-tuning on protest labels still pending |
| ReID integration | ✅ Done | Stable ID matching logic + logs |
| LLaVA description | ✅ Module done | Real LLaVA quality validation pending |
| Leader scoring | ✅ Done | Calibration on real labeled cases pending |
| Memory indexation | ✅ Done | Chroma semantic mode not fully validated in prod |
| RAG/chat interface | ✅ Done | CLI + API + Streamlit available |

---

## Runbook (Main Commands)

### Full multi-camera run
```bash
python src/pipeline/run_multicam_pipeline.py \
  --videos-dir <folder_with_camera_videos> \
  --run-name protest_day1 \
  --yolo-model yolov8n.pt \
  --osnet-ckpt checkpoints/osnet_market1501_gpu_full.pt \
  --descriptor-backend mock \
  --memory-backend simple \
  --question "who is likely the leader" \
  --device cuda
```

### API
```bash
python -m uvicorn src.interface.api:app --host 127.0.0.1 --port 8010
```

### Streamlit
```bash
python -m streamlit run src/interface/streamlit_app.py --server.port 8501
```

---

## What Remains (High Impact)

1. Fine-tune YOLO on protest-specific labeled dataset.
2. Validate LLaVA backend quality on real videos (not mock).
3. Add CI pipeline + formal unit/integration tests beyond smoke.
4. Finalize security/compliance docs (JWT policy, RGPD/AI Act notes, retention).
5. Update final report/slides with measured metrics and demo script.
