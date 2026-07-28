"""Tests for CellCommService response-mapping and error handling."""

import pytest

from src.api.cellcomm_service import CellCommService
from src.network.cellcomm import CellCommResult, InteractionResult


def make_service() -> CellCommService:
    return CellCommService()


def make_result() -> CellCommResult:
    return CellCommResult(
        cell_types=["T_cell", "B_cell"],
        cell_counts=[100, 100],
        n_cells=200,
        n_genes_tested=4,
        n_permutations=200,
        pvalue_threshold=0.05,
        n_pairs_tested=2,
        interactions=[
            InteractionResult(
                ligand="CD40LG", receptor="CD40",
                source_cluster="T_cell", target_cluster="B_cell",
                ligand_mean_expression=9.5, receptor_mean_expression=8.8,
                interaction_score=9.15, p_value=0.005,
            ),
        ],
    )


class TestResponseMapping:
    """Test cases for mapping a CellCommResult onto the API response schema."""

    def test_nodes_and_edges_map_correctly(self):
        service = make_service()
        response = service._to_response(make_result(), "pbmc_demo", "cell_type")

        assert response.dataset_source == "pbmc_demo"
        assert response.cell_type_key == "cell_type"
        assert response.n_cells == 200
        assert response.n_cell_types == 2
        assert response.n_significant == 1
        assert response.n_pairs_tested == 2

        assert {n.id for n in response.nodes} == {"T_cell", "B_cell"}
        t_cell_node = next(n for n in response.nodes if n.id == "T_cell")
        assert t_cell_node.n_cells == 100

        assert len(response.edges) == 1
        edge = response.edges[0]
        assert edge.source == "T_cell"
        assert edge.target == "B_cell"
        assert edge.ligand == "CD40LG"
        assert edge.receptor == "CD40"
        assert edge.p_value == pytest.approx(0.005)

    def test_no_significant_interactions_yields_empty_edges(self):
        service = make_service()
        result = make_result()
        result.interactions = []
        response = service._to_response(result, "pbmc_demo", "cell_type")

        assert response.edges == []
        assert response.n_significant == 0


class TestDemoDatasetErrors:
    """Test cases for the demo-dataset lookup error path."""

    @pytest.mark.asyncio
    async def test_unknown_dataset_raises_value_error(self):
        service = make_service()
        with pytest.raises(ValueError, match="Unknown demo dataset"):
            await service.infer_demo(
                "not_a_real_dataset", "cell_type",
                n_permutations=50, pvalue_threshold=0.05, min_cells_per_type=10,
            )


class TestListDemoDatasets:
    """Test cases for listing bundled demo datasets."""

    def test_lists_pbmc_demo(self):
        service = make_service()
        response = service.list_demo_datasets()
        names = [d.name for d in response.datasets]
        assert "pbmc_demo" in names
        pbmc = next(d for d in response.datasets if d.name == "pbmc_demo")
        assert pbmc.n_cells == 400
        assert set(pbmc.cell_types) == {"T_cell", "B_cell", "NK_cell", "Monocyte"}
