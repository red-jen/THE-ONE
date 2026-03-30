from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
import sys

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pipeline.run_multicam_pipeline import run_multicam_pipeline


def _sanitize(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", text.strip())
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned.strip("_") or "run"


def _save_uploaded_files(files, run_name: str) -> list[str]:
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    upload_dir = PROJECT_ROOT / "outputs" / "uploads" / f"streamlit_{_sanitize(run_name)}_{stamp}"
    upload_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for file in files:
        target = upload_dir / Path(file.name).name
        with target.open("wb") as f:
            f.write(file.getbuffer())
        saved.append(str(target))
    return saved


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    st.set_page_config(page_title="Human Behaviour Multicam", page_icon="🎥", layout="wide")

    st.title("🎥 Human Behaviour — Multi-camera Pipeline")
    st.caption("Upload protest camera videos, run full pipeline, and inspect outputs in one place.")

    with st.sidebar:
        st.header("Run Settings")
        run_name = st.text_input("Run name", value="streamlit_run")
        question = st.text_input("Question (optional)", value="who is most suspicious")

        st.subheader("Models")
        yolo_model = st.text_input("YOLO weights", value="yolov8n.pt")
        osnet_ckpt = st.text_input("OSNet checkpoint", value="checkpoints/osnet_market1501_gpu_full.pt")

        st.subheader("Tracking")
        det_conf = st.slider("Detection confidence", 0.0, 1.0, 0.25, 0.01)
        det_iou = st.slider("Detection IoU", 0.0, 1.0, 0.45, 0.01)
        sim_threshold = st.slider("ReID similarity threshold", 0.0, 1.0, 0.55, 0.01)
        max_age = st.number_input("Max track age (frames)", min_value=1, max_value=500, value=60)

        st.subheader("Description & Memory")
        descriptor_backend = st.selectbox("Descriptor backend", ["mock", "auto", "llava"], index=0)
        memory_backend = st.selectbox("Memory backend", ["simple", "auto", "chroma"], index=0)
        rag_generator_backend = st.selectbox("RAG generator", ["template", "ollama"], index=0)
        rag_ollama_model = st.text_input("Ollama model", value="llama3.1:8b")
        rag_ollama_url = st.text_input("Ollama URL", value="http://127.0.0.1:11434")
        sample_every = st.number_input("Sample every N frames", min_value=1, max_value=60, value=5)
        max_events = st.number_input("Max events per camera (0 = no limit)", min_value=0, max_value=50000, value=0)

        st.subheader("Runtime")
        device = st.selectbox("Device", ["cuda", "cpu"], index=0)

    uploaded_files = st.file_uploader(
        "Drop one or more camera videos",
        type=["mp4", "avi", "mov", "mkv", "webm", "m4v"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        st.success(f"{len(uploaded_files)} file(s) ready")
        st.write([f.name for f in uploaded_files])

    run_clicked = st.button("Run Full Pipeline", type="primary", use_container_width=True)

    if run_clicked:
        if not uploaded_files:
            st.error("Please upload at least one video file.")
            return

        run_name_clean = _sanitize(run_name)
        output_root = PROJECT_ROOT / "outputs" / "multicam"

        with st.spinner("Saving uploads and running pipeline... this can take time."):
            try:
                saved_paths = _save_uploaded_files(uploaded_files, run_name_clean)
                run_multicam_pipeline(
                    videos=saved_paths,
                    videos_dir=None,
                    run_name=run_name_clean,
                    output_root=str(output_root),
                    yolo_model=yolo_model,
                    osnet_ckpt=osnet_ckpt,
                    sim_threshold=float(sim_threshold),
                    max_age=int(max_age),
                    det_conf=float(det_conf),
                    det_iou=float(det_iou),
                    device=device,
                    descriptor_backend=descriptor_backend,
                    llava_model_id="llava-hf/llava-1.5-7b-hf",
                    strict_llava=False,
                    sample_every=int(sample_every),
                    max_events=None if int(max_events) == 0 else int(max_events),
                    score_top_k=5,
                    memory_backend=memory_backend,
                    question=question.strip() or None,
                    top_leaders=3,
                    top_evidence=5,
                    rag_generator_backend=rag_generator_backend,
                    rag_ollama_model=rag_ollama_model,
                    rag_ollama_url=rag_ollama_url,
                    rag_ollama_timeout_sec=60.0,
                )
            except Exception as exc:
                st.error(f"Pipeline failed: {exc}")
                return

        run_dir = output_root / run_name_clean
        manifest_path = run_dir / "run_manifest.json"

        if not manifest_path.exists():
            st.error("Pipeline finished but manifest file was not found.")
            return

        manifest = _read_json(manifest_path)

        st.success("Pipeline completed successfully")
        st.write(f"Run directory: {run_dir}")

        c1, c2, c3 = st.columns(3)
        c1.metric("Videos", int(manifest.get("videos_count", 0)))
        c2.metric("Merged rows", int(manifest.get("merged_rows", 0)))
        c3.metric("Run", manifest.get("run_name", run_name_clean))

        st.subheader("Manifest")
        st.json(manifest)

        rag_path = manifest.get("rag_answer_json")
        if rag_path and Path(rag_path).exists():
            st.subheader("RAG Answer")
            rag_payload = _read_json(Path(rag_path))
            st.text_area("Answer", rag_payload.get("answer", ""), height=220)
            with st.expander("Raw answer JSON"):
                st.json(rag_payload)

        st.subheader("Output shortcuts")
        st.write(f"- {run_dir / 'suspicion_scores_summary.json'}")
        st.write(f"- {run_dir / 'memory' / 'index_summary.json'}")
        st.write(f"- {run_dir / 'merged_descriptions.jsonl'}")


if __name__ == "__main__":
    main()
