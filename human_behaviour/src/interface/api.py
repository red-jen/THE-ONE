from __future__ import annotations

import importlib
import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from .chatbot import answer_query

try:
    from src.pipeline.run_multicam_pipeline import run_multicam_pipeline
except Exception:
    from pipeline.run_multicam_pipeline import run_multicam_pipeline


try:
    _fastapi_mod = importlib.import_module("fastapi")
    FastAPI = getattr(_fastapi_mod, "FastAPI")
    HTTPException = getattr(_fastapi_mod, "HTTPException")
    File = getattr(_fastapi_mod, "File")
    Form = getattr(_fastapi_mod, "Form")
    UploadFile = getattr(_fastapi_mod, "UploadFile")
    Depends = getattr(_fastapi_mod, "Depends")
    Security = getattr(_fastapi_mod, "Security")
except Exception as exc:
    raise RuntimeError("FastAPI is required to use src/interface/api.py") from exc

try:
    _fastapi_security_mod = importlib.import_module("fastapi.security")
    HTTPBearer = getattr(_fastapi_security_mod, "HTTPBearer")
    HTTPAuthorizationCredentials = getattr(_fastapi_security_mod, "HTTPAuthorizationCredentials")
except Exception as exc:
    raise RuntimeError("fastapi.security is required to use JWT auth in src/interface/api.py") from exc

try:
    _pydantic_mod = importlib.import_module("pydantic")
    BaseModel = getattr(_pydantic_mod, "BaseModel")
except Exception as exc:
    raise RuntimeError("Pydantic is required to use src/interface/api.py") from exc


DEFAULT_SCORES = os.getenv("HB_SCORES_JSONL", "outputs/scores/test_run_scores.jsonl")
DEFAULT_STORE = os.getenv("HB_SIMPLE_STORE", "outputs/memory/test_run/simple_store.json")
UPLOADS_ROOT = Path(os.getenv("HB_UPLOADS_ROOT", "outputs/uploads")).resolve()
MULTICAM_ROOT = Path(os.getenv("HB_MULTICAM_ROOT", "outputs/multicam")).resolve()

AUTH_ENABLED = os.getenv("HB_API_AUTH", "false").strip().lower() in {"1", "true", "yes", "on"}
JWT_SECRET = os.getenv("HB_JWT_SECRET", "change-me")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = int(os.getenv("HB_JWT_EXPIRE_MIN", "120"))
API_USERNAME = os.getenv("HB_API_USERNAME", "admin")
API_PASSWORD = os.getenv("HB_API_PASSWORD", "admin")

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
bearer_scheme = HTTPBearer(auto_error=False)


class AskRequest(BaseModel):
    question: str
    top_leaders: int = 3
    top_evidence: int = 5
    scores_jsonl: str | None = None
    simple_store: str | None = None


class TokenRequest(BaseModel):
    username: str
    password: str


class RunPathRequest(BaseModel):
    run_name: str
    videos: list[str] = []
    videos_dir: str | None = None
    question: str | None = None
    descriptor_backend: str = "mock"
    memory_backend: str = "simple"
    device: str = "cuda"
    det_conf: float = 0.25
    det_iou: float = 0.45
    sim_threshold: float = 0.55
    max_age: int = 60
    sample_every: int = 5
    max_events: int | None = None
    top_leaders: int = 3
    top_evidence: int = 5
    yolo_model: str = "yolov8n.pt"
    osnet_ckpt: str = "checkpoints/osnet_market1501_gpu_full.pt"


app = FastAPI(title="Human Behaviour Query API", version="0.1.0")


def _get_jwt_module() -> Any:
    try:
        return importlib.import_module("jwt")
    except Exception as exc:
        raise HTTPException(status_code=500, detail="JWT library is not installed") from exc


def _create_token(subject: str) -> str:
    jwt_mod = _get_jwt_module()
    now = datetime.now(timezone.utc)
    payload = {
        "sub": subject,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=JWT_EXPIRE_MINUTES)).timestamp()),
    }
    return jwt_mod.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def _decode_token(token: str) -> dict:
    jwt_mod = _get_jwt_module()
    try:
        decoded = jwt_mod.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        if not isinstance(decoded, dict):
            raise HTTPException(status_code=401, detail="Invalid token payload")
        return decoded
    except Exception as exc:
        raise HTTPException(status_code=401, detail=f"Invalid or expired token: {exc}") from exc


def require_auth(credentials: Any = Security(bearer_scheme)) -> dict:
    if not AUTH_ENABLED:
        return {"sub": "anonymous", "auth_enabled": False}

    if credentials is None or not credentials.credentials:
        raise HTTPException(status_code=401, detail="Missing bearer token")

    decoded = _decode_token(credentials.credentials)
    decoded["auth_enabled"] = True
    return decoded


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "scores_exists": Path(DEFAULT_SCORES).exists(),
        "store_exists": Path(DEFAULT_STORE).exists(),
        "auth_enabled": AUTH_ENABLED,
    }


@app.post("/auth/token")
def create_access_token(req: TokenRequest) -> dict:
    if not AUTH_ENABLED:
        raise HTTPException(status_code=400, detail="Auth is disabled. Set HB_API_AUTH=true to enable JWT.")

    if req.username != API_USERNAME or req.password != API_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = _create_token(subject=req.username)
    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in_minutes": JWT_EXPIRE_MINUTES,
    }


@app.post("/ask")
def ask(req: AskRequest, _: dict = Depends(require_auth)) -> dict:
    scores_path = req.scores_jsonl or DEFAULT_SCORES
    store_path = req.simple_store or DEFAULT_STORE

    if not Path(scores_path).exists():
        raise HTTPException(status_code=400, detail=f"Scores file not found: {scores_path}")
    if not Path(store_path).exists():
        raise HTTPException(status_code=400, detail=f"Simple store file not found: {store_path}")

    result = answer_query(
        question=req.question,
        scores_jsonl=scores_path,
        simple_store_json=store_path,
        top_leaders=req.top_leaders,
        top_evidence=req.top_evidence,
    )
    return result


@app.post("/run-multicam-path")
def run_multicam_path(req: RunPathRequest, _: dict = Depends(require_auth)) -> dict:
    if not req.videos and not req.videos_dir:
        raise HTTPException(status_code=400, detail="Provide videos and/or videos_dir")

    run_multicam_pipeline(
        videos=req.videos,
        videos_dir=req.videos_dir,
        run_name=req.run_name,
        output_root=str(MULTICAM_ROOT),
        yolo_model=req.yolo_model,
        osnet_ckpt=req.osnet_ckpt,
        sim_threshold=req.sim_threshold,
        max_age=req.max_age,
        det_conf=req.det_conf,
        det_iou=req.det_iou,
        device=req.device,
        descriptor_backend=req.descriptor_backend,
        llava_model_id="llava-hf/llava-1.5-7b-hf",
        strict_llava=False,
        sample_every=req.sample_every,
        max_events=req.max_events,
        score_top_k=5,
        memory_backend=req.memory_backend,
        question=req.question,
        top_leaders=req.top_leaders,
        top_evidence=req.top_evidence,
    )

    manifest = MULTICAM_ROOT / req.run_name / "run_manifest.json"
    if not manifest.exists():
        raise HTTPException(status_code=500, detail="Pipeline completed but manifest is missing")

    return {
        "status": "ok",
        "run_name": req.run_name,
        "manifest": str(manifest),
    }


@app.post("/run-multicam-upload")
async def run_multicam_upload(
    files: list[Any] = File(...),
    run_name: str = Form(...),
    question: Optional[str] = Form(None),
    descriptor_backend: str = Form("mock"),
    memory_backend: str = Form("simple"),
    device: str = Form("cuda"),
    det_conf: float = Form(0.25),
    det_iou: float = Form(0.45),
    sim_threshold: float = Form(0.55),
    max_age: int = Form(60),
    sample_every: int = Form(5),
    max_events: Optional[int] = Form(None),
    top_leaders: int = Form(3),
    top_evidence: int = Form(5),
    yolo_model: str = Form("yolov8n.pt"),
    osnet_ckpt: str = Form("checkpoints/osnet_market1501_gpu_full.pt"),
    _: dict = Depends(require_auth),
) -> dict:
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    upload_dir = UPLOADS_ROOT / f"{run_name}_{stamp}"
    upload_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[str] = []
    for file in files:
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in VIDEO_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"Unsupported video extension: {file.filename}")

        target = upload_dir / Path(file.filename).name
        with target.open("wb") as out_f:
            shutil.copyfileobj(file.file, out_f)
        saved_paths.append(str(target))

    run_multicam_pipeline(
        videos=saved_paths,
        videos_dir=None,
        run_name=run_name,
        output_root=str(MULTICAM_ROOT),
        yolo_model=yolo_model,
        osnet_ckpt=osnet_ckpt,
        sim_threshold=sim_threshold,
        max_age=max_age,
        det_conf=det_conf,
        det_iou=det_iou,
        device=device,
        descriptor_backend=descriptor_backend,
        llava_model_id="llava-hf/llava-1.5-7b-hf",
        strict_llava=False,
        sample_every=sample_every,
        max_events=max_events,
        score_top_k=5,
        memory_backend=memory_backend,
        question=question,
        top_leaders=top_leaders,
        top_evidence=top_evidence,
    )

    manifest = MULTICAM_ROOT / run_name / "run_manifest.json"
    if not manifest.exists():
        raise HTTPException(status_code=500, detail="Pipeline completed but manifest is missing")

    return {
        "status": "ok",
        "run_name": run_name,
        "uploaded_files": saved_paths,
        "manifest": str(manifest),
    }
