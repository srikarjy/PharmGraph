"""Service layer for the cell-cell communication network inference endpoint.

Wraps the pure `src.network.infer_cell_communication` algorithm, mapping its
plain-dataclass result onto the API's pydantic response models. Unlike
`GraphExplorerService` (which swallows flaky external-API failures into empty
200 responses), there is no external dependency here -- bad input is a real
validation error and is raised, not silently swallowed, so the endpoint layer
can turn it into a 400.
"""

import asyncio
from typing import Dict, List

from ..config import get_logger
from ..network import (
    LIGAND_RECEPTOR_PAIRS,
    CellCommResult,
    generate_demo_dataset,
    infer_cell_communication,
)
from .schemas import (
    CellCommDemoDataset,
    CellCommDemoDatasetsResponse,
    CellCommEdge,
    CellCommInferResponse,
    CellCommNode,
)

logger = get_logger(__name__)

# Registry of bundled demo datasets: name -> description.
_DEMO_DATASETS: Dict[str, str] = {
    "pbmc_demo": (
        "Synthetic, biologically-plausible immune-cell-like dataset (not real "
        "patient data): 400 cells across T_cell/B_cell/NK_cell/Monocyte, with a "
        "few well-known ligand-receptor signals planted on top of background noise."
    ),
}


class CellCommService:
    """Runs cell-cell communication inference and builds API response objects."""

    def list_demo_datasets(self) -> CellCommDemoDatasetsResponse:
        """List the bundled demo datasets available to the `dataset` param."""
        datasets: List[CellCommDemoDataset] = []
        for name, description in _DEMO_DATASETS.items():
            adata = generate_demo_dataset()
            datasets.append(CellCommDemoDataset(
                name=name,
                description=description,
                n_cells=adata.n_obs,
                n_genes=adata.n_vars,
                cell_types=sorted(adata.obs["cell_type"].unique().tolist()),
            ))
        return CellCommDemoDatasetsResponse(datasets=datasets)

    async def infer_demo(
        self,
        dataset: str,
        cell_type_key: str,
        n_permutations: int,
        pvalue_threshold: float,
        min_cells_per_type: int,
    ) -> CellCommInferResponse:
        """Run inference against a bundled demo dataset.

        Raises:
            ValueError: if `dataset` is not a known demo dataset name, or the
                inference input is otherwise invalid.
        """
        if dataset not in _DEMO_DATASETS:
            raise ValueError(
                f"Unknown demo dataset '{dataset}'; known datasets: {sorted(_DEMO_DATASETS)}"
            )
        adata = generate_demo_dataset()
        return await self._infer(
            adata, dataset, cell_type_key, n_permutations, pvalue_threshold, min_cells_per_type
        )

    async def infer_uploaded(
        self,
        adata,
        cell_type_key: str,
        n_permutations: int,
        pvalue_threshold: float,
        min_cells_per_type: int,
    ) -> CellCommInferResponse:
        """Run inference against a caller-uploaded AnnData object."""
        return await self._infer(
            adata, "uploaded", cell_type_key, n_permutations, pvalue_threshold, min_cells_per_type
        )

    async def _infer(
        self,
        adata,
        dataset_source: str,
        cell_type_key: str,
        n_permutations: int,
        pvalue_threshold: float,
        min_cells_per_type: int,
    ) -> CellCommInferResponse:
        # CPU-bound; keep the event loop free even though it's typically fast.
        result: CellCommResult = await asyncio.to_thread(
            infer_cell_communication,
            adata,
            cell_type_key,
            LIGAND_RECEPTOR_PAIRS,
            n_permutations,
            pvalue_threshold,
            min_cells_per_type,
        )
        return self._to_response(result, dataset_source, cell_type_key)

    @staticmethod
    def _to_response(
        result: CellCommResult, dataset_source: str, cell_type_key: str
    ) -> CellCommInferResponse:
        nodes = [
            CellCommNode(id=cell_type, label=cell_type, n_cells=count)
            for cell_type, count in zip(result.cell_types, result.cell_counts)
        ]
        edges = [
            CellCommEdge(
                id=f"{i.source_cluster}:{i.target_cluster}:{i.ligand}:{i.receptor}",
                source=i.source_cluster, target=i.target_cluster,
                ligand=i.ligand, receptor=i.receptor,
                ligand_mean_expression=i.ligand_mean_expression,
                receptor_mean_expression=i.receptor_mean_expression,
                interaction_score=i.interaction_score, p_value=i.p_value,
            )
            for i in result.interactions
        ]
        return CellCommInferResponse(
            dataset_source=dataset_source,
            cell_type_key=cell_type_key,
            n_cells=result.n_cells,
            n_genes_tested=result.n_genes_tested,
            n_cell_types=len(result.cell_types),
            n_permutations=result.n_permutations,
            pvalue_threshold=result.pvalue_threshold,
            nodes=nodes,
            edges=edges,
            n_pairs_tested=result.n_pairs_tested,
            n_significant=result.n_significant,
        )
