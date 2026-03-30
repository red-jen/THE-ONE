"""FastAPI backend for Protest Leader Detection.

Endpoints:
  Auth:
    POST /auth/token              — get JWT access token
  Health:
    GET  /health                  — service health + DB status
  Pipeline:
    POST /pipeline/run            — run pipeline on local video paths
    POST /pipeline/upload         — upload videos and run pipeline
  Analysis:
    POST /analyze/{run_id}        — run RAG scoring on a completed run
    POST /ask/{run_id}            — ask a question about a run
  Results:
    GET  /runs                    — list all pipeline runs
    GET  /runs/{run_id}           — get full results for a run
    GET  /runs/{run_id}/persons   — get scored persons for a run
    GET  /runs/{run_id}/queries   — get Q&A history for a run
"""

from __future__ import annotations

import importlib
import json
import os
import shutil
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional, Callable

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import env_bootstrap  # noqa: F401, E402 — load .env before config getenv

from scoring.rag_scorer import load_description_rows, score_with_llm
from interface.chatbot import answer_query
from database.connection import get_db, init_db, engine
from database import crud
from database.models import Base


def _get_pipeline_runner():
    """Lazy-import the pipeline to avoid loading heavy ML deps at startup."""
    from pipeline.run_multicam_pipeline import run_multicam_pipeline
    return run_multicam_pipeline

_fa = importlib.import_module("fastapi")
FastAPI = getattr(_fa, "FastAPI")
HTTPException = getattr(_fa, "HTTPException")
Form = getattr(_fa, "Form")
Request = getattr(_fa, "Request")
Depends = getattr(_fa, "Depends")
Security = getattr(_fa, "Security")
Response = getattr(_fa, "Response")

_fa_resp = importlib.import_module("fastapi.responses")
JSONResponse = getattr(_fa_resp, "JSONResponse")
FileResponse = getattr(_fa_resp, "FileResponse")

_fa_sec = importlib.import_module("fastapi.security")
HTTPBearer = getattr(_fa_sec, "HTTPBearer")

_pydantic = importlib.import_module("pydantic")
BaseModel = getattr(_pydantic, "BaseModel")

from sqlalchemy.orm import Session
from prometheus_client import Counter, Histogram, Gauge, REGISTRY, generate_latest, CONTENT_TYPE_LATEST
from starlette.datastructures import UploadFile as StarletteUploadFile

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _env_first_nonempty(*keys: str, default: str) -> str:
    """First set key wins (common .env names + HB_* aliases)."""
    for key in keys:
        val = os.getenv(key)
        if val is not None and str(val).strip() != "":
            return str(val).strip()
    return default


OUTPUTS_ROOT = Path(os.getenv("HB_OUTPUTS_ROOT", "outputs")).resolve()
MULTICAM_ROOT = Path(os.getenv("HB_MULTICAM_ROOT", str(OUTPUTS_ROOT / "multicam"))).resolve()
UPLOADS_ROOT = Path(os.getenv("HB_UPLOADS_ROOT", str(OUTPUTS_ROOT / "uploads"))).resolve()

AUTH_ENABLED = os.getenv("HB_API_AUTH", "true").strip().lower() in {"1", "true", "yes"}
JWT_SECRET = _env_first_nonempty("HB_JWT_SECRET", "SECRET_KEY", default="change-me-in-production")
JWT_ALGORITHM = _env_first_nonempty("HB_JWT_ALGORITHM", "ALGORITHM", default="HS256")
JWT_EXPIRE_MIN = int(_env_first_nonempty("HB_JWT_EXPIRE_MIN", "ACCESS_TOKEN_EXPIRE_MINUTES", default="120"))
API_USERNAME = os.getenv("HB_API_USERNAME", "admin")
API_PASSWORD = os.getenv("HB_API_PASSWORD", "admin")
API_USERS_JSON = os.getenv(
    "HB_API_USERS_JSON",
    '{"admin":{"password":"admin","role":"admin"},'
    '"analyst":{"password":"analyst","role":"analyst"},'
    '"viewer":{"password":"viewer","role":"viewer"}}',
)

OLLAMA_MODEL = os.getenv("HB_OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_URL = os.getenv("HB_OLLAMA_URL", "http://127.0.0.1:11434")

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
bearer_scheme = HTTPBearer(auto_error=False)


def _max_multipart_part_bytes() -> int:
    """Starlette defaults max_part_size to 1 MiB; raise for large video uploads."""
    mb = int(os.getenv("HB_MAX_MULTIPART_PART_MB", "512"))
    return max(1, mb) * 1024 * 1024


def _form_scalar_text(form: Any, key: str, default: Optional[str] = None) -> Optional[str]:
    v = form.get(key)
    if v is None:
        return default
    if isinstance(v, StarletteUploadFile):
        return default
    s = str(v).strip()
    return s if s else default


def _form_required_text(form: Any, key: str) -> str:
    t = _form_scalar_text(form, key)
    if not t:
        raise HTTPException(400, f"Missing or empty field: {key}")
    return t


def _form_float(form: Any, key: str, default: float) -> float:
    t = _form_scalar_text(form, key, None)
    if t is None:
        return default
    try:
        return float(t)
    except ValueError:
        raise HTTPException(400, f"Invalid number for {key}") from None


def _form_int(form: Any, key: str, default: Optional[int]) -> Optional[int]:
    t = _form_scalar_text(form, key, None)
    if t is None:
        return default
    try:
        return int(t)
    except ValueError:
        raise HTTPException(400, f"Invalid integer for {key}") from None

try:
    REQUEST_COUNT = Counter(
        "hb_http_requests_total",
        "Total HTTP requests",
        ["method", "endpoint", "status_code"],
    )
except ValueError:
    REQUEST_COUNT = REGISTRY._names_to_collectors["hb_http_requests"]  # type: ignore[attr-defined]

try:
    REQUEST_LATENCY = Histogram(
        "hb_http_request_duration_seconds",
        "HTTP request latency in seconds",
        ["method", "endpoint"],
        buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
    )
except ValueError:
    REQUEST_LATENCY = REGISTRY._names_to_collectors["hb_http_request_duration_seconds"]  # type: ignore[attr-defined]

try:
    IN_PROGRESS = Gauge(
        "hb_http_requests_in_progress",
        "HTTP requests in progress",
    )
except ValueError:
    IN_PROGRESS = REGISTRY._names_to_collectors["hb_http_requests_in_progress"]  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Protest Leader Detection API",
    description="AI pipeline: detect, track, describe, and score protest leaders using RAG + LLM",
    version="2.0.0",
)

try:
    _cors = importlib.import_module("fastapi.middleware.cors")
    CORSMiddleware = getattr(_cors, "CORSMiddleware")
    _cors_raw = os.getenv("HB_CORS_ORIGINS", "").strip()
    _cors_list = [o.strip() for o in _cors_raw.split(",") if o.strip()]
    if not _cors_list:
        _cors_list = ["*"]
    _cors_creds = (
        os.getenv("HB_CORS_CREDENTIALS", "").strip().lower() in {"1", "true", "yes"}
        and "*" not in _cors_list
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_list,
        allow_credentials=_cors_creds,
        allow_methods=["*"],
        allow_headers=["*"],
    )
except Exception:
    pass


@app.on_event("startup")
def on_startup():
    init_db()


@app.middleware("http")
async def metrics_middleware(request, call_next):
    method = request.method
    endpoint = request.url.path
    IN_PROGRESS.inc()
    start = time.perf_counter()
    status_code = "500"
    try:
        response = await call_next(request)
        status_code = str(response.status_code)
        return response
    finally:
        elapsed = time.perf_counter() - start
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status_code=status_code).inc()
        REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(elapsed)
        IN_PROGRESS.dec()


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

def _jwt_mod():
    try:
        return importlib.import_module("jwt")
    except Exception as exc:
        raise HTTPException(500, "PyJWT not installed") from exc


def _load_users() -> dict[str, dict[str, str]]:
    try:
        users = json.loads(API_USERS_JSON)
        if isinstance(users, dict):
            return users
    except Exception:
        pass
    return {API_USERNAME: {"password": API_PASSWORD, "role": "admin"}}


USERS = _load_users()


def _create_token(subject: str, role: str) -> str:
    jwt = _jwt_mod()
    now = datetime.now(timezone.utc)
    return jwt.encode(
        {"sub": subject, "role": role, "iat": int(now.timestamp()),
         "exp": int((now + timedelta(minutes=JWT_EXPIRE_MIN)).timestamp())},
        JWT_SECRET, algorithm=JWT_ALGORITHM,
    )


def _decode_token(token: str) -> dict:
    jwt = _jwt_mod()
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except Exception as exc:
        raise HTTPException(401, f"Invalid or expired token: {exc}") from exc


def require_auth(creds: Any = Security(bearer_scheme)) -> dict:
    if not AUTH_ENABLED:
        return {"sub": "anonymous", "role": "admin", "auth_enabled": False}
    if creds is None or not creds.credentials:
        raise HTTPException(401, "Missing bearer token")
    decoded = _decode_token(creds.credentials)
    if "role" not in decoded:
        decoded["role"] = "viewer"
    decoded["auth_enabled"] = True
    return decoded


def require_roles(*allowed_roles: str) -> Callable[..., dict]:
    allowed = set(allowed_roles)

    def _dep(user: dict = Depends(require_auth)) -> dict:
        if not AUTH_ENABLED:
            return user
        role = str(user.get("role", "viewer"))
        if role not in allowed:
            raise HTTPException(
                status_code=403,
                detail=f"Insufficient role. Required one of: {sorted(allowed)}; current: {role}",
            )
        return user

    return _dep


def _find_person_crop_file(manifest: dict, person_id: int) -> Path | None:
    """Find one saved crop image for a person in run outputs."""
    camera_runs = manifest.get("camera_runs", []) if isinstance(manifest, dict) else []
    track_id_part = f"track_{int(person_id):04d}"
    for cam in camera_runs:
        track_dir = cam.get("tracking_dir")
        if not track_dir:
            continue
        crops_dir = Path(track_dir) / "crops"
        if not crops_dir.exists():
            continue
        matches = sorted(crops_dir.glob(f"*{track_id_part}*.jpg"))
        if matches:
            return matches[0]
    return None


def _as_manifest_dict(raw_manifest: Any) -> dict:
    if isinstance(raw_manifest, dict):
        return raw_manifest
    if isinstance(raw_manifest, str):
        try:
            parsed = json.loads(raw_manifest)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class TokenRequest(BaseModel):
    username: str
    password: str


class AnalyzeRequest(BaseModel):
    ollama_model: str = OLLAMA_MODEL
    ollama_url: str = OLLAMA_URL
    timeout_sec: float = 120.0
    top_k: int = 10


class AskRequest(BaseModel):
    question: str
    top_leaders: int = 5
    top_evidence: int = 8
    generator_backend: str = "ollama"
    ollama_model: str = OLLAMA_MODEL
    ollama_url: str = OLLAMA_URL
    ollama_timeout_sec: float = 60.0


class PipelineRunRequest(BaseModel):
    run_name: str
    videos: list[str] = []
    videos_dir: str | None = None
    question: str | None = "Who is the most likely protest leader and why?"
    descriptor_backend: str = "mock"
    memory_backend: str = "simple"
    device: str = "cuda"
    yolo_model: str = "yolov8n.pt"
    osnet_ckpt: str = "checkpoints/osnet_market1501_gpu_full.pt"
    det_conf: float = 0.25
    det_iou: float = 0.45
    sim_threshold: float = 0.55
    max_age: int = 60
    sample_every: int = 5
    max_events: int | None = None
    top_leaders: int = 5
    top_evidence: int = 8
    ollama_model: str = OLLAMA_MODEL
    ollama_url: str = OLLAMA_URL


# ---------------------------------------------------------------------------
# Auth endpoint
# ---------------------------------------------------------------------------

@app.post("/auth/token", tags=["auth"])
def create_token(req: TokenRequest):
    if not AUTH_ENABLED:
        raise HTTPException(400, "Auth disabled. Set HB_API_AUTH=true.")
    user = USERS.get(req.username)
    if user is None or req.password != str(user.get("password", "")):
        raise HTTPException(401, "Invalid credentials")
    role = str(user.get("role", "viewer"))
    return {
        "access_token": _create_token(req.username, role),
        "token_type": "bearer",
        "expires_in_minutes": JWT_EXPIRE_MIN,
        "role": role,
    }


@app.get("/auth/me", tags=["auth"])
def auth_me(user: dict = Depends(require_auth)):
    return {
        "username": user.get("sub", "anonymous"),
        "role": user.get("role", "viewer"),
        "auth_enabled": bool(user.get("auth_enabled", AUTH_ENABLED)),
    }


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health", tags=["health"])
def health(db: Session = Depends(get_db)):
    # Keep this endpoint non-blocking: avoid running DB queries here.
    db_file_exists = False
    try:
        url = str(engine.url)
        if url.startswith("sqlite:///"):
            db_path = Path(url.replace("sqlite:///", "")).resolve()
            db_file_exists = db_path.exists()
    except Exception:
        db_file_exists = False
    return {
        "status": "ok",
        "version": "2.0.0",
        "database": "connected" if db_file_exists else "configured",
        "auth_enabled": AUTH_ENABLED,
        "ollama_url": OLLAMA_URL,
        "scoring_backend": "rag_llm",
    }


@app.get("/metrics", tags=["health"])
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ---------------------------------------------------------------------------
# Pipeline endpoints
# ---------------------------------------------------------------------------

@app.post("/pipeline/run", tags=["pipeline"])
def pipeline_run(
    req: PipelineRunRequest,
    db: Session = Depends(get_db),
    _: dict = Depends(require_roles("admin", "analyst")),
):
    if not req.videos and not req.videos_dir:
        raise HTTPException(400, "Provide videos and/or videos_dir")

    run_record = crud.create_run(db, run_name=req.run_name, videos_count=len(req.videos))

    try:
        _get_pipeline_runner()(
            videos=req.videos, videos_dir=req.videos_dir,
            run_name=req.run_name, output_root=str(MULTICAM_ROOT),
            yolo_model=req.yolo_model, osnet_ckpt=req.osnet_ckpt,
            sim_threshold=req.sim_threshold, max_age=req.max_age,
            det_conf=req.det_conf, det_iou=req.det_iou, device=req.device,
            descriptor_backend=req.descriptor_backend,
            llava_model_id="llava-hf/llava-1.5-7b-hf", strict_llava=False,
            sample_every=req.sample_every, max_events=req.max_events,
            score_top_k=req.top_leaders, memory_backend=req.memory_backend,
            question=req.question, top_leaders=req.top_leaders,
            top_evidence=req.top_evidence, rag_generator_backend="template",
            rag_ollama_model=req.ollama_model, rag_ollama_url=req.ollama_url,
            rag_ollama_timeout_sec=5.0,
        )

        manifest_path = MULTICAM_ROOT / req.run_name / "run_manifest.json"
        manifest = {}
        if manifest_path.exists():
            with manifest_path.open("r", encoding="utf-8") as f:
                manifest = json.load(f)

        merged = manifest.get("merged_descriptions")
        if merged and Path(merged).exists():
            crud.import_descriptions_from_jsonl(db, run_record.id, merged)

        leader_jsonl = manifest.get("leader_scores_jsonl")
        if leader_jsonl and Path(leader_jsonl).exists():
            scores = []
            with Path(leader_jsonl).open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        scores.append(json.loads(line))
            crud.save_leader_scores(
                db, run_record.id, scores,
                manifest.get("leader_scoring_backend", "unknown"),
            )

        crud.complete_run(
            db, run_record.id, status="completed",
            manifest=manifest,
            scoring_backend=manifest.get("leader_scoring_backend"),
        )

    except Exception as exc:
        crud.complete_run(db, run_record.id, status="failed", error_message=str(exc))
        raise HTTPException(500, f"Pipeline failed: {exc}") from exc

    return {
        "status": "ok",
        "run_id": run_record.id,
        "run_name": req.run_name,
    }


@app.post("/pipeline/upload", tags=["pipeline"])
async def pipeline_upload(
    request: Request,
    db: Session = Depends(get_db),
    _: dict = Depends(require_roles("admin", "analyst")),
):
    form = await request.form(max_part_size=_max_multipart_part_bytes())
    try:
        raw_files = form.getlist("files")
        # request.form() yields starlette.datastructures.UploadFile, not fastapi.UploadFile (different types).
        files = [x for x in raw_files if isinstance(x, StarletteUploadFile)]
        if not files:
            raise HTTPException(400, "No files uploaded")

        run_name = _form_required_text(form, "run_name")
        question = _form_scalar_text(form, "question") or "Who is the most likely protest leader?"
        descriptor_backend = _form_scalar_text(form, "descriptor_backend") or "mock"
        memory_backend = _form_scalar_text(form, "memory_backend") or "simple"
        device = _form_scalar_text(form, "device") or "cuda"
        det_conf = _form_float(form, "det_conf", 0.25)
        det_iou = _form_float(form, "det_iou", 0.45)
        sim_threshold = _form_float(form, "sim_threshold", 0.55)
        max_age = _form_int(form, "max_age", 60)
        if max_age is None:
            max_age = 60
        sample_every = _form_int(form, "sample_every", 5)
        if sample_every is None:
            sample_every = 5
        max_events = _form_int(form, "max_events", None)
        top_leaders = _form_int(form, "top_leaders", 5)
        if top_leaders is None:
            top_leaders = 5
        top_evidence = _form_int(form, "top_evidence", 8)
        if top_evidence is None:
            top_evidence = 8
        yolo_model = _form_scalar_text(form, "yolo_model") or "yolov8n.pt"
        osnet_ckpt = _form_scalar_text(form, "osnet_ckpt") or "checkpoints/osnet_market1501_gpu_full.pt"
        ollama_model = _form_scalar_text(form, "ollama_model") or OLLAMA_MODEL
        ollama_url = _form_scalar_text(form, "ollama_url") or OLLAMA_URL

        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        upload_dir = UPLOADS_ROOT / f"{run_name}_{stamp}"
        upload_dir.mkdir(parents=True, exist_ok=True)

        saved: list[str] = []
        for file in files:
            suffix = Path(file.filename or "").suffix.lower()
            if suffix not in VIDEO_EXTENSIONS:
                raise HTTPException(400, f"Unsupported format: {file.filename}")
            target = upload_dir / Path(file.filename).name
            with target.open("wb") as out_f:
                shutil.copyfileobj(file.file, out_f)
            saved.append(str(target))

        run_record = crud.create_run(db, run_name=run_name, videos_count=len(saved))

        for i, path in enumerate(saved, 1):
            crud.add_video(db, run_record.id, f"cam_{i}", Path(path).name, path)

        try:
            _get_pipeline_runner()(
                videos=saved, videos_dir=None,
                run_name=run_name, output_root=str(MULTICAM_ROOT),
                yolo_model=yolo_model, osnet_ckpt=osnet_ckpt,
                sim_threshold=sim_threshold, max_age=max_age,
                det_conf=det_conf, det_iou=det_iou, device=device,
                descriptor_backend=descriptor_backend,
                llava_model_id="llava-hf/llava-1.5-7b-hf", strict_llava=False,
                sample_every=sample_every, max_events=max_events,
                score_top_k=top_leaders, memory_backend=memory_backend,
                question=question, top_leaders=top_leaders,
                top_evidence=top_evidence, rag_generator_backend="template",
                rag_ollama_model=ollama_model, rag_ollama_url=ollama_url,
                rag_ollama_timeout_sec=5.0,
            )

            manifest_path = MULTICAM_ROOT / run_name / "run_manifest.json"
            manifest = {}
            if manifest_path.exists():
                with manifest_path.open("r", encoding="utf-8") as f:
                    manifest = json.load(f)

            merged = manifest.get("merged_descriptions")
            if merged and Path(merged).exists():
                crud.import_descriptions_from_jsonl(db, run_record.id, merged)

            leader_jsonl = manifest.get("leader_scores_jsonl")
            if leader_jsonl and Path(leader_jsonl).exists():
                scores = []
                with Path(leader_jsonl).open("r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            scores.append(json.loads(line))
                crud.save_leader_scores(
                    db, run_record.id, scores,
                    manifest.get("leader_scoring_backend", "unknown"),
                )

            crud.complete_run(db, run_record.id, status="completed", manifest=manifest)

        except Exception as exc:
            crud.complete_run(db, run_record.id, status="failed", error_message=str(exc))
            raise HTTPException(500, f"Pipeline failed: {exc}") from exc

        return {"status": "ok", "run_id": run_record.id, "run_name": run_name, "uploaded_files": saved}
    finally:
        await form.close()


# ---------------------------------------------------------------------------
# Analysis endpoints
# ---------------------------------------------------------------------------

@app.post("/analyze/{run_id}", tags=["analysis"])
def analyze(
    run_id: int,
    req: AnalyzeRequest,
    db: Session = Depends(get_db),
    _: dict = Depends(require_roles("admin", "analyst")),
):
    run = crud.get_run(db, run_id)
    if not run:
        raise HTTPException(404, "Run not found")

    manifest = _as_manifest_dict(run.manifest)
    desc_path = manifest.get("merged_descriptions")
    if not desc_path or not Path(desc_path).exists():
        raise HTTPException(400, "No descriptions available for this run")

    rows = load_description_rows(desc_path)
    result = score_with_llm(
        description_rows=rows, model=req.ollama_model,
        ollama_url=req.ollama_url, timeout_sec=req.timeout_sec, top_k=req.top_k,
    )

    crud.save_leader_scores(db, run_id, result["candidates"], result["backend_used"])

    return {
        "status": "ok",
        "run_id": run_id,
        "scoring_backend": result["backend_used"],
        "model": result["model"],
        "candidates": result["candidates"],
    }


@app.post("/ask/{run_id}", tags=["analysis"])
def ask(run_id: int, req: AskRequest, db: Session = Depends(get_db), _: dict = Depends(require_auth)):
    run = crud.get_run(db, run_id)
    if not run:
        raise HTTPException(404, "Run not found")

    manifest = _as_manifest_dict(run.manifest)
    scores_path = manifest.get("leader_scores_jsonl", manifest.get("scores_jsonl", ""))
    store_dir = manifest.get("memory_dir", "")
    store_path = str(Path(store_dir) / "simple_store.json") if store_dir else ""

    if not scores_path or not Path(scores_path).exists():
        raise HTTPException(400, "No scores available. Run /analyze first.")
    if not store_path or not Path(store_path).exists():
        raise HTTPException(400, "No memory store available.")

    result = answer_query(
        question=req.question, scores_jsonl=scores_path,
        simple_store_json=store_path, top_leaders=req.top_leaders,
        top_evidence=req.top_evidence, generator_backend=req.generator_backend,
        ollama_model=req.ollama_model, ollama_url=req.ollama_url,
        ollama_timeout_sec=req.ollama_timeout_sec,
    )

    crud.save_query(
        db, run_id=run_id, question=req.question,
        answer=result.get("answer", ""),
        generator_backend=result.get("generator_backend", ""),
        model_used=result.get("model"),
        evidence_count=len(result.get("evidence_used", [])),
    )

    return result


# ---------------------------------------------------------------------------
# Results endpoints
# ---------------------------------------------------------------------------

@app.get("/runs", tags=["results"])
def list_runs(limit: int = 50, offset: int = 0, db: Session = Depends(get_db), _: dict = Depends(require_auth)):
    runs = crud.list_runs(db, limit=limit, offset=offset)
    return [
        {
            "id": r.id,
            "run_name": r.run_name,
            "status": r.status,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "completed_at": r.completed_at.isoformat() if r.completed_at else None,
            "videos_count": r.videos_count,
            "scoring_backend": r.scoring_backend,
        }
        for r in runs
    ]


@app.get("/runs/{run_id}", tags=["results"])
def get_run(run_id: int, db: Session = Depends(get_db), _: dict = Depends(require_auth)):
    run = crud.get_run(db, run_id)
    if not run:
        raise HTTPException(404, "Run not found")

    persons = crud.get_persons_for_run(db, run_id)
    scores = crud.get_leader_scores(db, run_id)
    queries = crud.get_queries(db, run_id)

    return {
        "id": run.id,
        "run_name": run.run_name,
        "status": run.status,
        "created_at": run.created_at.isoformat() if run.created_at else None,
        "completed_at": run.completed_at.isoformat() if run.completed_at else None,
        "videos_count": run.videos_count,
        "scoring_backend": run.scoring_backend,
        "error_message": run.error_message,
        "persons": [
            {
                "person_id": p.person_id,
                "crop_url": f"/runs/{run.id}/persons/{p.person_id}/crop",
                "leader_score": p.leader_score,
                "heuristic_score": p.heuristic_score,
                "reasoning": p.reasoning,
                "observations": p.observations,
                "cameras_seen": p.cameras_seen,
                "duration_sec": p.duration_sec,
                "front_count": p.front_count,
                "megaphone_count": p.megaphone_count,
                "banner_count": p.banner_count,
                "gesture_total": p.gesture_total,
            }
            for p in persons
        ],
        "leader_scores": [
            {"person_id": s.person_id, "leader_score": s.leader_score,
             "reasoning": s.reasoning, "backend_used": s.backend_used}
            for s in scores
        ],
        "queries": [
            {"id": q.id, "question": q.question, "answer": q.answer,
             "asked_at": q.asked_at.isoformat() if q.asked_at else None}
            for q in queries
        ],
    }


@app.get("/runs/{run_id}/persons", tags=["results"])
def get_persons(run_id: int, db: Session = Depends(get_db), _: dict = Depends(require_auth)):
    run = crud.get_run(db, run_id)
    if not run:
        raise HTTPException(404, "Run not found")
    persons = crud.get_persons_for_run(db, run_id)
    return [
        {
            "person_id": p.person_id,
            "crop_url": f"/runs/{run.id}/persons/{p.person_id}/crop",
            "leader_score": p.leader_score,
            "heuristic_score": p.heuristic_score,
            "reasoning": p.reasoning,
            "observations": p.observations,
            "cameras_seen": p.cameras_seen,
            "duration_sec": p.duration_sec,
            "front_count": p.front_count,
            "center_count": p.center_count,
            "back_count": p.back_count,
            "megaphone_count": p.megaphone_count,
            "banner_count": p.banner_count,
            "flag_count": p.flag_count,
            "microphone_count": p.microphone_count,
            "gesture_total": p.gesture_total,
            "scoring_backend": p.scoring_backend,
        }
        for p in persons
    ]


@app.get("/runs/{run_id}/persons/{person_id}/crop", tags=["results"])
def get_person_crop(
    run_id: int,
    person_id: int,
    db: Session = Depends(get_db),
    _: dict = Depends(require_auth),
):
    run = crud.get_run(db, run_id)
    if not run:
        raise HTTPException(404, "Run not found")
    manifest = _as_manifest_dict(run.manifest)
    crop_path = _find_person_crop_file(manifest, person_id)
    if crop_path is None or not crop_path.exists():
        raise HTTPException(404, "Crop image not found for this person")
    return FileResponse(str(crop_path), media_type="image/jpeg")


@app.get("/runs/{run_id}/queries", tags=["results"])
def get_queries_endpoint(run_id: int, db: Session = Depends(get_db), _: dict = Depends(require_auth)):
    run = crud.get_run(db, run_id)
    if not run:
        raise HTTPException(404, "Run not found")
    queries = crud.get_queries(db, run_id)
    return [
        {
            "id": q.id,
            "question": q.question,
            "answer": q.answer,
            "generator_backend": q.generator_backend,
            "model_used": q.model_used,
            "evidence_count": q.evidence_count,
            "asked_at": q.asked_at.isoformat() if q.asked_at else None,
        }
        for q in queries
    ]
