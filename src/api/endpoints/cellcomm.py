"""API routes for cell-cell communication network inference."""

import os
import uuid
from typing import Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from ...config import get_config, get_logger
from ..cellcomm_service import CellCommService
from ..schemas import CellCommDemoDatasetsResponse, CellCommInferResponse

logger = get_logger(__name__)

router = APIRouter()

_service = CellCommService()


@router.get("/demo-datasets", response_model=CellCommDemoDatasetsResponse)
async def demo_datasets():
    """List the bundled synthetic demo datasets available to `/infer`."""
    return _service.list_demo_datasets()


@router.post("/infer", response_model=CellCommInferResponse)
async def infer(
    dataset: Optional[str] = Query(None, description="Name of a bundled demo dataset"),
    file: Optional[UploadFile] = File(None, description="A real .h5ad AnnData file"),
    cell_type_key: str = Query("cell_type", description="obs column holding cell-type labels"),
    n_permutations: int = Query(200, ge=10, le=1000, description="Number of label permutations"),
    pvalue_threshold: float = Query(0.05, gt=0.0, le=1.0, description="Significance threshold"),
    min_cells_per_type: int = Query(10, ge=1, description="Minimum cells required per cell type"),
):
    """Infer a ligand-receptor interaction network between cell types.

    Provide exactly one of `dataset` (a bundled demo dataset name) or `file`
    (an uploaded .h5ad AnnData file).
    """
    if (dataset is None) == (file is None):
        raise HTTPException(
            status_code=400, detail="Provide exactly one of `dataset` or `file`"
        )

    config = get_config().cellcomm
    if n_permutations > config.max_n_permutations:
        raise HTTPException(
            status_code=400,
            detail=f"n_permutations exceeds the server maximum of {config.max_n_permutations}",
        )

    try:
        if dataset is not None:
            return await _service.infer_demo(
                dataset, cell_type_key, n_permutations, pvalue_threshold, min_cells_per_type
            )
        adata = await _load_uploaded_anndata(file)
        return await _service.infer_uploaded(
            adata, cell_type_key, n_permutations, pvalue_threshold, min_cells_per_type
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


async def _load_uploaded_anndata(file: UploadFile):
    """Write an uploaded .h5ad to a temp file, load it, and clean up."""
    import anndata as ad

    temp_dir = get_config().temp_directory
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, f"cellcomm_upload_{uuid.uuid4().hex}.h5ad")
    try:
        contents = await file.read()
        with open(temp_path, "wb") as f:
            f.write(contents)
        return ad.read_h5ad(temp_path)
    except Exception as e:
        raise ValueError(f"Could not read uploaded file as an AnnData .h5ad: {e}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
