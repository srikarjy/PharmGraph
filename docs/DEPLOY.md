# Deploying PharmGraph

The graph explorer needs **no database and no API key** — Open Targets' GraphQL
API is public. It deploys as two pieces: a FastAPI service and a static React build.

## Option A — Render (one-click blueprint, recommended)

The repo includes [`render.yaml`](../render.yaml), which defines both services.

1. Push this repo to GitHub.
2. In [Render](https://render.com), click **New + → Blueprint** and select the repo.
3. Render creates `pharmgraph-api` (Python) and `pharmgraph-web` (static). Apply.
4. When `pharmgraph-api` is live, copy its URL (e.g. `https://pharmgraph-api.onrender.com`).
5. In `pharmgraph-web` → Environment, set `VITE_API_BASE` to that URL and redeploy.
6. Open the `pharmgraph-web` URL, search **CYP2C9**, and expand.

Both services run on Render's free tier. (Free web services sleep when idle and
take ~30s to wake on the first request — fine for a demo.)

## Option B — Split hosts (Vercel/Netlify frontend + any Python host)

**Backend** (Fly.io, Railway, Render, a VM — anything that runs Python):

```bash
pip install -r requirements.txt
uvicorn src.api.main:app --host 0.0.0.0 --port $PORT
# allow your frontend origin:
export PHARMGRAPH_CORS_ORIGINS="https://your-frontend.vercel.app"
```

**Frontend** (Vercel/Netlify): set project root to `frontend/`, build `npm run build`,
output `dist`, and set env `VITE_API_BASE=https://your-api-host`.

## Environment variables

| Variable | Where | Purpose |
| --- | --- | --- |
| `PHARMGRAPH_CORS_ORIGINS` | API | Comma-separated allowed origins (or `*` for a public demo). |
| `VITE_API_BASE` | Frontend build | Absolute URL of the deployed API. Empty in dev (Vite proxies `/api`). |
| `PHARMGRAPH_ENABLE_ML` | API | Optional; `1` loads the experimental ML stack. Leave unset for the graph explorer. |

## Local run

```bash
# Backend
pip install -r requirements.txt
uvicorn src.api.main:app --reload --port 8000

# Frontend (separate terminal) — Vite proxies /api to :8000
cd frontend && npm install && npm run dev
```
