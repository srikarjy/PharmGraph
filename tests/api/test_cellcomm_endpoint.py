"""Endpoint smoke tests for /api/v1/cellcomm -- the app has a history of
silently failing to boot end-to-end (see docs/gene_drug_protein_graph_blueprint.md
Sec 3), so this exercises the real FastAPI app rather than just the service layer.
"""

from fastapi.testclient import TestClient

from src.api.main import app


class TestCellCommEndpoints:
    """Test cases for the cell-cell communication router."""

    def test_demo_datasets_lists_pbmc_demo(self):
        with TestClient(app) as client:
            response = client.get("/api/v1/cellcomm/demo-datasets")
            assert response.status_code == 200
            body = response.json()
            names = [d["name"] for d in body["datasets"]]
            assert "pbmc_demo" in names

    def test_infer_against_demo_dataset(self):
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/cellcomm/infer",
                params={"dataset": "pbmc_demo", "n_permutations": 100},
            )
            assert response.status_code == 200
            body = response.json()
            assert body["n_significant"] >= 1
            assert {n["id"] for n in body["nodes"]} == {
                "T_cell", "B_cell", "NK_cell", "Monocyte",
            }
            assert body["edges"]
            assert all(e["p_value"] <= 0.05 for e in body["edges"])

    def test_infer_requires_exactly_one_source(self):
        with TestClient(app) as client:
            response = client.post("/api/v1/cellcomm/infer")
            assert response.status_code == 400

    def test_infer_rejects_unknown_dataset(self):
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/cellcomm/infer", params={"dataset": "not_a_real_dataset"}
            )
            assert response.status_code == 400
