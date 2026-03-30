# Docker + Prometheus + Grafana

## Services

- `frontend` -> React app on `http://localhost:3000`
- `api` -> FastAPI backend on `http://localhost:8010`
- `postgres` -> PostgreSQL on `localhost:5432`
- `prometheus` -> Metrics server on `http://localhost:9090`
- `grafana` -> Dashboards on `http://localhost:3001`
- `mlflow` -> Tracking server on `http://localhost:5000`

## Quick start

1. Copy env file:

```bash
cp .env.example .env
```

2. Start stack:

```bash
docker compose up -d --build
```

### If compose fails with “container name is already in use”

Old containers from a previous project folder or manual `docker run` can block fixed names. Remove them, then start again:

**PowerShell (Windows)**

```powershell
docker rm -f mlflow-server hb-prometheus hb-grafana human-behaviour-api human-behaviour-frontend hb-postgres 2>$null
docker compose -f docker-compose.yml up -d --build
```

Or use `.\scripts\docker-stack-up.ps1` from this folder.

3. Validate:

- API health: `http://localhost:8010/health`
- API metrics: `http://localhost:8010/metrics`
- Prometheus targets: `http://localhost:9090/targets`
- Grafana: `http://localhost:3001` (default `admin/admin`)

## Grafana dashboard

The dashboard is auto-provisioned:

- `Human Behaviour API Overview`
- `Stack health (MLflow note + Prometheus)` — explains MLflow vs metrics; shows `up{}` for API and Prometheus

It includes:

- Requests per second
- P95 latency
- In-progress requests
- Request counts by endpoint and status

## MLflow vs Prometheus

- **MLflow** stores runs, params, and artifacts. Open the UI at `http://localhost:5000`. The default MLflow server image **does not** expose a Prometheus scrape endpoint; Grafana dashboards for HTTP/metrics come from **FastAPI `/metrics`**.
- Optional: set `MLFLOW_TRACKING_URI=http://mlflow:5000` in the API container (already defaulted in `docker-compose.yml`) when you enable MLflow in pipeline code.

## Compose dependency order

- `postgres` → healthy, then **API** starts.
- **MLflow** healthy, then **API** starts (tracking server is ready).
- **API** healthy, then **Prometheus** starts (targets can scrape immediately).
- **Prometheus** up, then **Grafana** starts.
