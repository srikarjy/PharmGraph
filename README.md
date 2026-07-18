# PharmGraph

Search a gene, protein, or drug and explore its real pharmacogenomic interaction
network — sourced live from [Open Targets](https://platform.opentargets.org/),
deduplicated, and ranked by [CPIC](https://cpicpgx.org/) evidence level.

## Why

Open Targets returns pharmacogenomics data as hundreds of duplicated rows per gene
(one per allele, genotype, and study), with no evidence ranking and no visual
surface. PharmGraph aggregates that into a clean, interactive graph.

For **CYP2C9**, that means 343 raw rows become 35 unique drug interactions, each
keeping its strongest CPIC annotation.

## Features

- Live search over genes, proteins, and drugs
- Interactive force-directed interaction graph (click a node to expand it)
- Deduplication with best-evidence-per-pair aggregation
- Filter interactions by minimum CPIC evidence tier
- Supporting PubMed citations and PharmGKB clinical-annotation links on every interaction

## Run locally

No database or API key required — Open Targets' API is public.

**Backend**
```bash
pip install -r requirements.txt
uvicorn src.api.main:app --reload --port 8000
```

**Frontend** (separate terminal)
```bash
cd frontend
npm install
npm run dev
```

Open the printed URL, search `CYP2C9`, and expand the result.

## API

- `GET /api/v1/graph/search?q=CYP2C9`
- `GET /api/v1/graph/expand/gene/ENSG00000138109?min_evidence=2A`

Full docs at `http://localhost:8000/docs`.

## Tech stack

- **Backend:** Python, FastAPI, httpx, Pydantic
- **Frontend:** React 19, TypeScript, Vite, react-force-graph-2d
- **Data:** Open Targets GraphQL v4, PubMed

## Tests

```bash
pytest tests/api/test_graph_service.py
```

## Deploy

A [`render.yaml`](render.yaml) blueprint deploys the API and frontend together.
See [docs/DEPLOY.md](docs/DEPLOY.md).

## License

MIT
