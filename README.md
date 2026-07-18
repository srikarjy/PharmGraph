# PharmGraph

**An interactive gene ↔ protein ↔ drug interaction-graph explorer for pharmacogenomics, backed by live [Open Targets](https://platform.opentargets.org/) data.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![React 19](https://img.shields.io/badge/react-19-149eca.svg)](https://react.dev/)

Search a gene, protein, or drug and explore its real pharmacogenomic interaction
network — deduplicated from hundreds of raw clinical-evidence rows and ranked by
[CPIC](https://cpicpgx.org/) evidence level.

---

## What this solves

When a patient carries a variant like **CYP2C9**, the clinical question isn't
"does this one gene affect this one drug." It's *what is the full web of genes,
proteins, and drugs this variant touches, and how strong is the evidence for each
link?*

Open Targets holds that answer, but returns it as **hundreds of duplicated rows
per gene** — one per allele, genotype, and study — with no visual surface and no
evidence ranking. PharmGraph turns that raw, redundant data into a clean,
evidence-ranked, interactive graph.

### Before vs. after (CYP2C9, verified live)

| Dimension          | Raw Open Targets / toy lookup   | PharmGraph                                |
| ------------------ | ------------------------------- | ----------------------------------------- |
| Rows returned      | 343 raw rows, heavily duplicated| **35** unique drug nodes, deduplicated    |
| Protein identity   | 28 mixed-source protein IDs     | **1** canonical UniProt SwissProt node    |
| Evidence handling  | none — all rows weighted equally| CPIC-ranked, best annotation per pair     |
| Relationships shown| single pairwise gene → drug     | full gene ↔ protein ↔ drug network        |
| Surface            | raw JSON, no UI                 | interactive force-directed graph          |

The dedup is a ~90% row reduction (343 → 35 entities) that keeps the single
highest-confidence CPIC annotation per drug–gene pair. Figures above are for
CYP2C9; the reduction ratio varies by gene.

---

## Project status

This repository began as a broader pharmacogenomics platform. The **actively
maintained, verified, and demo-able component is the PharmGraph interaction-graph
explorer** described here. Other modules are experimental and in varying states of
completeness — see the honest breakdown below.

| Component | Path | Status |
| --- | --- | --- |
| **Graph explorer API** (Open Targets client, aggregation service, endpoints) | `src/api/graph_service.py`, `src/api/endpoints/graph.py`, `src/data_ingestion/opentargets_client.py` | ✅ Working, 19 unit tests, verified live end-to-end |
| **Graph explorer frontend** (React 19 + force-directed graph) | `frontend/` | ✅ Working, verified in-browser |
| **Core API** (FastAPI app, auth, rate limiting, monitoring) | `src/api/` | ✅ Boots and serves the graph explorer |
| NCBI/PubMed data ingestion | `src/data_ingestion/` | ⚠️ Mostly passing; a few tests failing |
| ML / AutoML pipeline & model serving | `src/ml/`, `src/api/model_service.py` | 🧪 Experimental, **opt-in** (heavy native deps); not required for the graph explorer |
| Analytics & reporting | `src/analytics/` | 🧪 Experimental; some test-collection errors |
| Clinical decision-support endpoints | referenced in `main.py` | 🚧 Scaffolded, not implemented |

The ML stack pulls in heavy native dependencies (torch/MLflow) that are not needed
by the graph explorer and can crash on import in some environments. It is therefore
**disabled by default** and gated behind `PHARMGRAPH_ENABLE_ML=1`.

---

## Quick start (graph explorer)

### Backend

```bash
# 1. Install runtime dependencies (no ML libs needed)
pip install -r requirements.txt

# 2. Run the API — the graph explorer is live at /api/v1/graph
uvicorn src.api.main:app --reload --port 8000
```

- Swagger UI: http://localhost:8000/docs
- Search:  `GET /api/v1/graph/search?q=CYP2C9`
- Expand:  `GET /api/v1/graph/expand/gene/ENSG00000138109`

No API key and no database are required — Open Targets' public GraphQL API is
unauthenticated.

### Frontend

```bash
cd frontend
npm install
npm run dev          # Vite dev server, proxies /api to :8000
```

Then open the printed URL, search **CYP2C9**, and expand the result to render its
live interaction network.

### Enabling the experimental ML stack (optional)

```bash
pip install -r requirements-ml.txt
PHARMGRAPH_ENABLE_ML=1 uvicorn src.api.main:app --port 8000
```

---

## How it works

Four stages, with the real work in stage 2:

1. **Fetch live** — `OpenTargetsClient`: an httpx GraphQL client with adaptive
   rate limiting and TTL response caching, hardened for GraphQL errors returned
   inside HTTP 200 responses.
2. **Aggregate** — `GraphExplorerService`: collapses hundreds of duplicated
   pharmacogenomics rows into deduplicated nodes and edges, keeping the single
   highest-confidence annotation per drug–gene pair.
3. **Rank by evidence** — maps the CPIC scale (1A → 4) to per-edge confidence
   scores, so every link carries a defensible strength.
4. **Render** — a React force-directed graph: search an entity, expand its
   network, click any node for detail.

### Tech stack

- **Backend:** Python, FastAPI, httpx, Pydantic
- **Frontend:** React 19, Vite, TypeScript, `react-force-graph-2d`
- **Data:** Open Targets Platform GraphQL v4 (live, unauthenticated)
- **Tooling:** pytest, Docker, GitHub Actions

---

## Testing

```bash
# Graph explorer service — the verified core
pytest tests/api/test_graph_service.py        # 19 passing

# Broader suite (includes experimental modules with known gaps)
pytest tests/api tests/config tests/data_ingestion
```

The graph explorer has been verified end-to-end against live Open Targets data
(searching CYP2C9 expands to a gene, its canonical protein, and its interacting
drugs — including warfarin) and in-browser via Playwright.

---

## License

MIT — see [LICENSE](LICENSE).

## Author

Srikar J · srikarjy@bu.edu · Boston University
