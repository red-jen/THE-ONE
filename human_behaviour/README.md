# Human Behaviour — Protest leader analysis stack

End-to-end pipeline: **YOLO** detection → **OSNet** re-identification & tracking → **LLaVA** (or mock) descriptions → **heuristic + LLM (Ollama)** leadership scoring → **FastAPI** + **React** UI. Optional: **Postgres**, **MLflow**, **Prometheus**, **Grafana** via Docker.

## Requirements

- **Docker Desktop** (for full stack), or Python 3.12 + Node 22 for local dev
- **OSNet checkpoint** for real Re-ID: place under `checkpoints/osnet_market1501_gpu_full.pt` (not in Git — copy from your old machine or train; see `src/training/`)
- **Ollama** on the host for LLM scoring/chat (Docker API uses `http://host.docker.internal:11434` by default)

## Quick start (Docker — recommended)

```bash
cd human_behaviour
cp .env.example .env
# Edit .env: HB_JWT_SECRET, Grafana password, optional POSTGRES_* / HB_OLLAMA_URL

docker compose up -d --build
```

| URL | Service |
|-----|---------|
| http://localhost:3000 | Frontend (nginx) |
| http://localhost:8010/docs | FastAPI |
| http://localhost:5433 | Postgres **on host** (container name `postgres`, internal port 5432) |
| http://localhost:5000 | MLflow |
| http://localhost:9090 | Prometheus |
| http://localhost:3001 | Grafana |

**Login (default):** e.g. `admin` / `admin` if using the sample `HB_API_USERS_JSON` in compose.

**Windows / port 5432 conflict:** Compose publishes Postgres as **5433→5432** so a local PostgreSQL can keep 5432.

**Stop:** `docker compose down` (add `-v` to drop volumes).

**Full reset of named containers:** `powershell -File scripts/docker-stack-up.ps1`

## Local development (no Docker API)

```bash
cd human_behaviour
pip install -r requirements.api-docker.txt   # or requirements.txt for full ML stack
# SQLite default; or set DATABASE_URL in .env for Postgres
python -m uvicorn src.interface.api:app --host 127.0.0.1 --port 8000 --reload
```

```bash
cd human_behaviour/frontend
npm ci
npm run dev
```

Frontend talks to the API at `http://127.0.0.1:8000` in dev by default (see `src/lib/api.js`). For Docker API on the host at 8010, use `frontend/.env.local`:

```env
VITE_API_ORIGIN=http://127.0.0.1:8010
```

## Project layout

- `src/interface/api.py` — FastAPI app
- `src/pipeline/run_multicam_pipeline.py` — orchestration
- `src/tracking/reid_tracker.py` — YOLO + OSNet tracking, crops under `outputs/.../tracking/crops/`
- `src/description/llava_descriptor.py` — per-person **crop** → caption
- `src/scoring/rag_scorer.py` — Ollama leadership judge + heuristic fallback
- `frontend/` — Vite + React UI
- `docker-compose.yml` — postgres, mlflow, api, frontend, prometheus, grafana
- `docs/uml/` — Mermaid diagrams

## Tests & CI

```bash
cd human_behaviour
pip install -r requirements.api-docker.txt pytest httpx
pytest -q tests
```

GitHub Actions: `.github/workflows/ci.yml` (backend tests, frontend build, Docker image build checks).

## Documentation

- `DOCKER_MONITORING.md` — monitoring ports and notes
- `Architecture.md` — high-level architecture (if present)

## License / data

Do not commit `.env`, large videos, `outputs/`, or `checkpoints/`; they are gitignored. Copy checkpoints and `.env` securely when moving machines.
