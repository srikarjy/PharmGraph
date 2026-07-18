"""API routes for gene/protein/drug interaction graph exploration."""

import asyncio
from typing import Optional

from fastapi import APIRouter, Query, HTTPException

from ...config import get_logger
from ...data_ingestion.opentargets_client import OpenTargetsClient
from ..graph_service import GraphExplorerService
from ..schemas import GraphSearchResponse, GraphExpandResponse

logger = get_logger(__name__)

router = APIRouter()

# Lazily-initialized, process-wide OpenTargetsClient. Entered once (its underlying
# httpx.AsyncClient is opened) on first request and kept alive for the process lifetime,
# since this is a public read-only demo API with no per-request session semantics.
# Assumes a single, persistent event loop for the app's lifetime (true under uvicorn).
# Note for tests: FastAPI's TestClient must be used as a context manager
# (`with TestClient(app) as client:`) so repeated calls share one event loop --
# without it, each call gets a fresh loop and this singleton silently breaks.
_client: Optional[OpenTargetsClient] = None
_client_lock = asyncio.Lock()


async def get_graph_service() -> GraphExplorerService:
    """FastAPI dependency providing a GraphExplorerService backed by a shared client."""
    global _client
    if _client is None:
        async with _client_lock:
            if _client is None:
                client = OpenTargetsClient.from_config()
                await client.__aenter__()
                _client = client
    return GraphExplorerService(_client)


@router.get("/search", response_model=GraphSearchResponse)
async def search(
    q: str = Query(..., min_length=1, description="Gene symbol, protein, or drug name"),
    limit: int = Query(10, ge=1, le=25, description="Maximum number of candidates"),
):
    """Search for a gene, protein-coding target, or drug by name."""
    service = await get_graph_service()
    return await service.search(q, limit)


CPIC_LEVELS = ("1A", "1B", "2A", "2B", "3", "4")


@router.get("/expand/{node_type}/{node_id}", response_model=GraphExpandResponse)
async def expand(
    node_type: str,
    node_id: str,
    limit: int = Query(15, ge=1, le=40, description="Maximum number of interaction partners"),
    min_evidence: Optional[str] = Query(
        None,
        description="Minimum CPIC evidence tier to include (1A strongest .. 4 weakest)",
    ),
):
    """Expand a gene or drug node into its pharmacogenomic interaction subgraph."""
    if node_type not in ("gene", "target", "drug"):
        raise HTTPException(status_code=400, detail="node_type must be 'gene' or 'drug'")
    if min_evidence is not None and min_evidence not in CPIC_LEVELS:
        raise HTTPException(
            status_code=400,
            detail=f"min_evidence must be one of {', '.join(CPIC_LEVELS)}",
        )

    service = await get_graph_service()
    return await service.expand(node_type, node_id, limit, min_evidence)
