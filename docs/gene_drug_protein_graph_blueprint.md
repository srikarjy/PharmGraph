# Gene ↔ Protein ↔ Drug Graph Explorer — Blueprint

Status: implementation in progress. This file is the working reference for the build — resolved facts, exact schema, and decisions. Update it as tasks complete; it supersedes assumptions in the original plan file where they conflict (the plan guessed field names; this document has them confirmed live).

## 1. Problem

This repo is backend-only with no working demo surface (see "Known breakage" below). The feature: a user searches a gene, protein, or drug and interactively explores its real pharmacogenomic interaction network — sourced live from Open Targets, not a toy pairwise lookup.

## 2. Data source — confirmed live via GraphQL introspection

Public, unauthenticated API: `https://api.platform.opentargets.org/api/v4/graphql`. No API key, no rate limit encountered from direct curl (the earlier rate limit was on Claude's MCP wrapper tool, not the underlying API).

### `Target.pharmacogenomics` / `Drug.pharmacogenomics`
Both entity types expose this field. **No pagination arguments — always returns the full list.** Confirmed for CYP2C9 (`ENSG00000138109`): **343 rows, 35 unique drugs**, heavy duplication (many rows per drug across different genotypes/variants/phenotypes).

Returns `[Pharmacogenomics!]!`. Real fields (introspected, not guessed):

| Field | Type | Notes |
|---|---|---|
| `genotypeId` | String, nullable | often `null`; when present looks like `10_94947445_C_C,T` |
| `genotypeAnnotationText` | String, nullable | |
| `phenotypeText` | String, nullable | human-readable, e.g. `"decreased risk of over-anticoagulation"` |
| `pgxCategory` | String, nullable | observed values: `dosage`, `toxicity`, `efficacy`, `metabolism/pk`, `other` |
| `evidenceLevel` | String, nullable | observed values: `1A`, `1B`, `2A`, `3`, `4` (CPIC scale; `2B` not seen in this sample but expect it exists) |
| `isDirectTarget` | Boolean, non-null | |
| `target` | `Target` object, nullable | embedded — full target, not just an id |
| `drugs` | `[DrugWithIdentifiers!]!` | **a list**, not a single drug — one pgx row can name multiple drugs |
| `literature` | `[String!]`, nullable | PubMed IDs |
| `studyId`, `datasourceId`, `datatypeId`, `variantId`, `variantRsId`, `haplotypeId`, `haplotypeFromSourceId`, `variantFunctionalConsequenceId` | String, nullable | not needed for v1 |

`DrugWithIdentifiers` fields: `drugId` (String), `drugFromSource` (String), `drug` (embedded `Drug` object — `.id`, `.name` confirmed present).

### `Target.proteinIds`
`[IdAndSource!]!`, no pagination. For CYP2C9 returns **28 entries** across multiple sources (`uniprot_swissprot`, `uniprot_trembl`, `ensembl_PRO`, `uniprot_obsolete`). **Use exactly the entry where `source == "uniprot_swissprot"`** as the canonical protein node — CYP2C9 has exactly one (`P11712`). If a target has zero `uniprot_swissprot` entries (rare edge case), skip the protein node rather than guessing.

### Sample real rows (CYP2C9 → warfarin, verified live)
```json
{"genotypeId": null, "phenotypeText": "decreased risk of over-anticoagulation",
 "pgxCategory": "toxicity", "evidenceLevel": "1A", "isDirectTarget": false,
 "drugs": [{"drugId": "CHEMBL1464", "drug": {"id": "CHEMBL1464", "name": "WARFARIN"}}]}
```
This confirms the textbook CYP2C9↔warfarin case this whole platform is themed around actually resolves correctly through this API path.

## 3. Known breakage (fixed)

The app had never successfully booted. Found and fixed, in order of discovery:

1. `config/config.yaml` did not exist. `ConfigManager._load_yaml_config()` raised `ConfigError`. Fixed: created `config/config.yaml`.
2. `main.py::_setup_middleware` called `self.config.get("api.cors.origins", ["*"])` — `AppConfig` is a plain dataclass, no `.get()`. Fixed: added `APIConfig` dataclass, updated call sites to `self.config.api.cors_origins`.
3. `src/api/auth.py::AuthManager.__init__` (instantiated at *module import time*, unconditionally) called `self.config.get("auth.jwt_secret", ...)` — same broken pattern, different file. Fixed: added `AuthConfig` dataclass, updated to `self.config.auth.jwt_secret`.
4. `AuthManager.__init__` also referenced `self.api_keys` inside `_create_default_users()` before `self.api_keys = {}` was assigned — pure ordering bug. Fixed: moved the assignment earlier.
5. `main.py`'s request-logging middleware and all three exception handlers called `logger.info/error/warning(msg, request_id=..., status_code=..., ...)` — passing arbitrary kwargs straight to a plain stdlib `logging.Logger` (confirmed via `src/config/logging_config.py`: `get_logger()` returns a stdlib logger with a custom `StructuredFormatter`/`ContextFilter`, not a structlog-bound logger, despite `main.py` separately calling `structlog.configure()` which doesn't actually wire into this project's `get_logger`). This meant **every single HTTP request would 500**. Fixed: wrapped the custom fields in `extra={...}` (the mechanism `StructuredFormatter` actually reads from `record.__dict__`), 6 call sites in `main.py`.

(Pre-existing, explicitly NOT touched) `main.py` imports `drug_gene, risk_scoring, dosing, clinical_support` from `src/api/endpoints/`, which don't exist as files — silently caught by `try/except ImportError`. The new graph router is wired independently so this doesn't block us.

**Verified end-to-end**: full `PharmacogenomicsAPI` app (with only the unrelated ML/transformers import chain stubbed out — a pre-existing, out-of-scope environment issue, not a code bug) boots, serves `/health` (200), `/docs` (200), `/api/v1/graph/search?q=CYP2C9` (200, correct top hit), `/api/v1/graph/expand/gene/ENSG00000138109` (200, 17 nodes/16 edges including warfarin), and 404s correctly on unknown routes.

## 4. Aggregation logic (gene → drug direction, `expand(gene, ensembl_id)`)

This is the one piece of real business logic in the feature — turning 343 noisy rows into ~15 clean edges:

1. Fetch `target.pharmacogenomics` (all rows) + `target.proteinIds`.
2. Build 1 gene node + 1 protein node (if a `uniprot_swissprot` id exists) + 1 `encodes` edge between them.
3. Flatten: for each pgx row, for each entry in `row.drugs` (a row can list >1 drug), emit a `(drugId, drugName, pgxCategory, evidenceLevel, phenotypeText)` tuple.
4. Group flattened tuples by `drugId`. Within a group, keep the row with the **highest evidence level** (rank `1A > 1B > 2A > 2B > 3 > 4 > null`), and count `annotation_count = len(group)` for the tooltip ("won out over 6 other annotations").
5. Sort groups by derived confidence descending, take top `limit` (default 15, max 40).
6. Emit one `drug` node + one `pgx_interaction` edge per surviving group. Edge carries `action_type=pgxCategory`, `phenotype=phenotypeText` (from the winning row), `evidence_level`, `confidence` (mapped), and `annotation_count`.

`expand(drug, chembl_id)` mirrors this exactly, grouping by `target.id` (gene) instead of `drugId`, using `Drug.pharmacogenomics`.

Confidence mapping (CPIC evidence level → float, documented constant, not derived from any external formula):
```python
EVIDENCE_CONFIDENCE = {"1A": 0.95, "1B": 0.85, "2A": 0.70, "2B": 0.60, "3": 0.40, "4": 0.25}
DEFAULT_CONFIDENCE = 0.30  # missing/unrecognized evidenceLevel
```

## 5. Architecture (unchanged from approved plan, see `~/.claude/plans/cosmic-stirring-muffin.md`)

- **Boot fix**: `config/config.yaml` (new) + `APIConfig` dataclass in `src/config/models.py` + fix `main.py`'s two `.get()` calls.
- **`OpenTargetsConfig`**: new dataclass, same shape as `NCBIConfig`, no auth fields.
- **`src/data_ingestion/opentargets_client.py`**: httpx-based, retry/backoff shape copied from `NCBIClient`, `AdaptiveRateLimiter` reused as-is, `cachetools.TTLCache` for in-memory response caching. Must check `body.get("errors")` on HTTP 200 — GraphQL errors don't always map to HTTP status codes.
- **Schemas** (`src/api/schemas.py`): `GraphNodeType`, `GraphEdgeType`, `GraphNode`, `GraphEdge`, `GraphSearchCandidate`, `GraphSearchResponse`, `GraphExpandResponse`.
- **Router + service**: `src/api/endpoints/graph.py` (`GET /search?q=`, `GET /expand/{node_type}/{node_id}`) + `src/api/graph_service.py::GraphExplorerService` (owns the aggregation logic in §4). Wired into `main.py` as its own `include_router` call, prefix `/api/v1/graph`, no auth.
- **Frontend** (`frontend/`): Vite + React + TS + `react-force-graph-2d`. Search replaces the graph; clicking a node merges new nodes/edges into existing state by id (never replaces the array) so the force simulation doesn't reset on expand.

## 6. Decisions (resolved)

- **Tests**: unit tests only, `tests/api/test_graph_service.py`, covering the aggregation logic (grouping, evidence-level ranking, limit capping). No endpoint/integration tests — kept in scope.
- **Node cap default**: 15 drug nodes on first expand.
- **`isDirectTarget`**: not surfaced in v1 (minor, revisit later if needed).

Everything else in the original plan stands as approved.

## 7. Task checklist

- [x] 1. Verify Open Targets schema (this document)
- [x] 2. Fix app boot (config.yaml + APIConfig, + auth.py bugs + logging middleware bug found along the way)
- [x] 3. Add OpenTargetsConfig
- [x] 4. Build OpenTargetsClient
- [x] 5. Add graph Pydantic schemas
- [x] 6. Build graph router + service (+ unit tests, 19 passing)
- [x] 7. Verify backend end-to-end (full app boot + real Open Targets data, 47 tests passing)
- [x] 8. Scaffold React frontend
- [x] 9. Browser click-through verification (Playwright, headless Chromium — search "CYP2C9" → real hit, expand → gene+protein+15 drug nodes incl. warfarin render on canvas, click gene node → tooltip/detail panel populate. Zero console errors, zero failed/4xx-5xx requests. Note: local sandbox has an unrelated process already bound to :8000 — backend was run on :8001 for this test via a temporary `vite.config.ts` proxy edit, reverted after.)
