"""Tests for GraphExplorerService pharmacogenomics aggregation logic."""

import pytest
from unittest.mock import AsyncMock

from src.api.graph_service import (
    GraphExplorerService, _evidence_rank, _confidence_for, _meets_min_evidence,
    EVIDENCE_CONFIDENCE, DEFAULT_CONFIDENCE, MAX_LITERATURE_PER_EDGE,
)
from src.api.schemas import GraphNodeType, GraphEdgeType
from src.data_ingestion.opentargets_client import OpenTargetsResponse


def make_service(client=None) -> GraphExplorerService:
    return GraphExplorerService(client or AsyncMock())


class TestEvidenceMapping:
    """Test cases for evidence level ranking/confidence helpers."""

    def test_confidence_for_known_levels(self):
        for level, expected in EVIDENCE_CONFIDENCE.items():
            assert _confidence_for(level) == expected

    def test_confidence_for_unknown_level_uses_default(self):
        assert _confidence_for("unknown") == DEFAULT_CONFIDENCE
        assert _confidence_for(None) == DEFAULT_CONFIDENCE

    def test_evidence_rank_orders_best_first(self):
        assert _evidence_rank("1A") < _evidence_rank("1B") < _evidence_rank("2A")
        assert _evidence_rank("2A") < _evidence_rank("2B") < _evidence_rank("3") < _evidence_rank("4")

    def test_evidence_rank_unknown_is_worst(self):
        assert _evidence_rank(None) > _evidence_rank("4")
        assert _evidence_rank("garbage") > _evidence_rank("4")


class TestMinEvidence:
    """Test cases for the min-evidence threshold helper."""

    def test_no_threshold_admits_everything(self):
        assert _meets_min_evidence("4", None) is True
        assert _meets_min_evidence(None, None) is True

    def test_threshold_admits_equal_or_stronger(self):
        assert _meets_min_evidence("1A", "2A") is True
        assert _meets_min_evidence("2A", "2A") is True

    def test_threshold_rejects_weaker(self):
        assert _meets_min_evidence("3", "2A") is False
        assert _meets_min_evidence("4", "1A") is False

    def test_unknown_level_rejected_when_threshold_set(self):
        assert _meets_min_evidence(None, "4") is False
        assert _meets_min_evidence("garbage", "4") is False


class TestCanonicalProteinId:
    """Test cases for picking the canonical UniProt id."""

    def test_picks_swissprot_entry(self):
        service = make_service()
        protein_ids = [
            {"id": "A0A2R8YF67", "source": "uniprot_trembl"},
            {"id": "P11712", "source": "uniprot_swissprot"},
            {"id": "ENSP00000260682", "source": "ensembl_PRO"},
        ]
        assert service._canonical_protein_id(protein_ids) == "P11712"

    def test_returns_none_when_no_swissprot_entry(self):
        service = make_service()
        protein_ids = [{"id": "A0A2R8YF67", "source": "uniprot_trembl"}]
        assert service._canonical_protein_id(protein_ids) is None

    def test_returns_none_for_empty_list(self):
        service = make_service()
        assert service._canonical_protein_id([]) is None


class TestGroupByDrug:
    """Test cases for aggregating gene-side pharmacogenomics rows by drug."""

    def test_keeps_best_evidence_level_per_drug(self):
        service = make_service()
        rows = [
            {"pgxCategory": "dosage", "phenotypeText": "lower dose", "evidenceLevel": "3",
             "drugs": [{"drugId": "CHEMBL1464", "drug": {"id": "CHEMBL1464", "name": "WARFARIN"}}]},
            {"pgxCategory": "toxicity", "phenotypeText": "decreased risk of over-anticoagulation",
             "evidenceLevel": "1A",
             "drugs": [{"drugId": "CHEMBL1464", "drug": {"id": "CHEMBL1464", "name": "WARFARIN"}}]},
        ]
        groups = service._group_by_drug(rows)

        assert len(groups) == 1
        group = groups["CHEMBL1464"]
        assert group["evidence_level"] == "1A"
        assert group["phenotype_text"] == "decreased risk of over-anticoagulation"
        assert group["pgx_category"] == "toxicity"
        assert group["count"] == 2

    def test_row_with_multiple_drugs_fans_out_to_multiple_groups(self):
        service = make_service()
        rows = [
            {"pgxCategory": "efficacy", "phenotypeText": "reduced response", "evidenceLevel": "2A",
             "drugs": [
                 {"drugId": "CHEMBL16", "drug": {"id": "CHEMBL16", "name": "PHENYTOIN"}},
                 {"drugId": "CHEMBL563", "drug": {"id": "CHEMBL563", "name": "FLURBIPROFEN"}},
             ]},
        ]
        groups = service._group_by_drug(rows)
        assert set(groups.keys()) == {"CHEMBL16", "CHEMBL563"}

    def test_rows_missing_drug_id_are_skipped(self):
        service = make_service()
        rows = [{"pgxCategory": "dosage", "evidenceLevel": "3", "drugs": [{"drugId": None, "drug": None}]}]
        assert service._group_by_drug(rows) == {}

    def test_falls_back_to_drug_id_when_name_missing(self):
        service = make_service()
        rows = [{"pgxCategory": "dosage", "evidenceLevel": "3",
                  "drugs": [{"drugId": "CHEMBL999", "drug": None}]}]
        groups = service._group_by_drug(rows)
        assert groups["CHEMBL999"]["drug_name"] == "CHEMBL999"


class TestGroupByGene:
    """Test cases for aggregating drug-side pharmacogenomics rows by gene."""

    def test_keeps_best_evidence_level_per_gene(self):
        service = make_service()
        rows = [
            {"pgxCategory": "dosage", "evidenceLevel": "4",
             "target": {"id": "ENSG00000138109", "approvedSymbol": "CYP2C9"}},
            {"pgxCategory": "toxicity", "evidenceLevel": "1A",
             "target": {"id": "ENSG00000138109", "approvedSymbol": "CYP2C9"}},
        ]
        groups = service._group_by_gene(rows)

        assert len(groups) == 1
        assert groups["ENSG00000138109"]["evidence_level"] == "1A"
        assert groups["ENSG00000138109"]["count"] == 2

    def test_rows_missing_target_are_skipped(self):
        service = make_service()
        rows = [{"pgxCategory": "dosage", "evidenceLevel": "3", "target": None}]
        assert service._group_by_gene(rows) == {}


class TestLiteratureAggregation:
    """Test cases for deduplicating PubMed IDs across aggregated rows."""

    def test_unions_pmids_across_rows_for_same_drug(self):
        service = make_service()
        rows = [
            {"pgxCategory": "dosage", "evidenceLevel": "3", "literature": ["111", "222"],
             "drugs": [{"drugId": "CHEMBL1", "drug": {"id": "CHEMBL1", "name": "D1"}}]},
            {"pgxCategory": "toxicity", "evidenceLevel": "1A", "literature": ["222", "333"],
             "drugs": [{"drugId": "CHEMBL1", "drug": {"id": "CHEMBL1", "name": "D1"}}]},
        ]
        groups = service._group_by_drug(rows)
        assert groups["CHEMBL1"]["literature"] == {"111", "222", "333"}

    def test_missing_literature_yields_empty_set(self):
        service = make_service()
        rows = [{"pgxCategory": "dosage", "evidenceLevel": "3",
                  "drugs": [{"drugId": "CHEMBL1", "drug": {"id": "CHEMBL1", "name": "D1"}}]}]
        assert service._group_by_drug(rows)["CHEMBL1"]["literature"] == set()


class TestExpandGene:
    """Test cases for GraphExplorerService.expand('gene', ...)."""

    @pytest.mark.asyncio
    async def test_builds_gene_protein_and_drug_nodes(self):
        client = AsyncMock()
        client.get_target_with_pharmacogenomics.return_value = OpenTargetsResponse(
            success=True,
            data={
                "target": {
                    "id": "ENSG00000138109",
                    "approvedSymbol": "CYP2C9",
                    "approvedName": "cytochrome P450 family 2 subfamily C member 9",
                    "proteinIds": [{"id": "P11712", "source": "uniprot_swissprot"}],
                    "pharmacogenomics": [
                        {"pgxCategory": "toxicity", "phenotypeText": "over-anticoagulation risk",
                         "evidenceLevel": "1A",
                         "drugs": [{"drugId": "CHEMBL1464", "drug": {"id": "CHEMBL1464", "name": "WARFARIN"}}]},
                    ],
                }
            },
        )
        service = make_service(client)

        result = await service.expand("gene", "ENSG00000138109", limit=15)

        node_types = {n.id: n.type for n in result.nodes}
        assert node_types["ENSG00000138109"] == GraphNodeType.GENE
        assert node_types["P11712"] == GraphNodeType.PROTEIN
        assert node_types["CHEMBL1464"] == GraphNodeType.DRUG

        relationships = {(e.source, e.target): e.relationship for e in result.edges}
        assert relationships[("ENSG00000138109", "P11712")] == GraphEdgeType.ENCODES
        assert relationships[("ENSG00000138109", "CHEMBL1464")] == GraphEdgeType.PGX_INTERACTION
        assert result.total_available == 1
        assert result.truncated is False

    @pytest.mark.asyncio
    async def test_respects_limit_and_sets_truncated(self):
        client = AsyncMock()
        pgx_rows = [
            {"pgxCategory": "metabolism/pk", "phenotypeText": f"effect {i}", "evidenceLevel": "3",
             "drugs": [{"drugId": f"CHEMBL{i}", "drug": {"id": f"CHEMBL{i}", "name": f"DRUG{i}"}}]}
            for i in range(5)
        ]
        client.get_target_with_pharmacogenomics.return_value = OpenTargetsResponse(
            success=True,
            data={"target": {"id": "ENSG1", "approvedSymbol": "GENE1", "approvedName": None,
                              "proteinIds": [], "pharmacogenomics": pgx_rows}},
        )
        service = make_service(client)

        result = await service.expand("gene", "ENSG1", limit=2)

        drug_nodes = [n for n in result.nodes if n.type == GraphNodeType.DRUG]
        assert len(drug_nodes) == 2
        assert result.total_available == 5
        assert result.truncated is True

    @pytest.mark.asyncio
    async def test_returns_empty_response_on_client_failure(self):
        client = AsyncMock()
        client.get_target_with_pharmacogenomics.return_value = OpenTargetsResponse(
            success=False, data=None, error_message="boom",
        )
        service = make_service(client)

        result = await service.expand("gene", "ENSG_BAD", limit=15)

        assert result.center_node_id == "ENSG_BAD"
        assert result.nodes == []
        assert result.edges == []

    @pytest.mark.asyncio
    async def test_unsupported_node_type_returns_empty_response_not_exception(self):
        service = make_service()
        result = await service.expand("protein", "P11712", limit=15)
        assert result.nodes == []
        assert result.edges == []

    @pytest.mark.asyncio
    async def test_target_alias_expands_like_gene(self):
        client = AsyncMock()
        client.get_target_with_pharmacogenomics.return_value = OpenTargetsResponse(
            success=True,
            data={"target": {"id": "ENSG1", "approvedSymbol": "GENE1", "approvedName": None,
                             "proteinIds": [], "pharmacogenomics": [
                                 {"pgxCategory": "dosage", "evidenceLevel": "1A", "literature": ["1"],
                                  "drugs": [{"drugId": "CHEMBL1", "drug": {"id": "CHEMBL1", "name": "D1"}}]}]}},
        )
        service = make_service(client)
        result = await service.expand("target", "ENSG1", limit=15)
        assert any(n.id == "CHEMBL1" for n in result.nodes)

    @pytest.mark.asyncio
    async def test_min_evidence_filters_weaker_interactions(self):
        client = AsyncMock()
        pgx_rows = [
            {"pgxCategory": "dosage", "evidenceLevel": "1A", "literature": [],
             "drugs": [{"drugId": "CHEMBL_STRONG", "drug": {"id": "CHEMBL_STRONG", "name": "STRONG"}}]},
            {"pgxCategory": "dosage", "evidenceLevel": "3", "literature": [],
             "drugs": [{"drugId": "CHEMBL_WEAK", "drug": {"id": "CHEMBL_WEAK", "name": "WEAK"}}]},
        ]
        client.get_target_with_pharmacogenomics.return_value = OpenTargetsResponse(
            success=True,
            data={"target": {"id": "ENSG1", "approvedSymbol": "GENE1", "approvedName": None,
                             "proteinIds": [], "pharmacogenomics": pgx_rows}},
        )
        service = make_service(client)

        result = await service.expand("gene", "ENSG1", limit=15, min_evidence="2A")

        drug_ids = {n.id for n in result.nodes if n.type == GraphNodeType.DRUG}
        assert drug_ids == {"CHEMBL_STRONG"}
        assert result.total_available == 1

    @pytest.mark.asyncio
    async def test_edges_carry_capped_sorted_literature(self):
        client = AsyncMock()
        many_pmids = [str(9000 + i) for i in range(MAX_LITERATURE_PER_EDGE + 5)]
        client.get_target_with_pharmacogenomics.return_value = OpenTargetsResponse(
            success=True,
            data={"target": {"id": "ENSG1", "approvedSymbol": "GENE1", "approvedName": None,
                             "proteinIds": [], "pharmacogenomics": [
                                 {"pgxCategory": "toxicity", "evidenceLevel": "1A", "literature": many_pmids,
                                  "drugs": [{"drugId": "CHEMBL1", "drug": {"id": "CHEMBL1", "name": "D1"}}]}]}},
        )
        service = make_service(client)

        result = await service.expand("gene", "ENSG1", limit=15)

        pgx_edge = next(e for e in result.edges if e.target == "CHEMBL1")
        assert len(pgx_edge.literature) == MAX_LITERATURE_PER_EDGE
        assert pgx_edge.literature == sorted(pgx_edge.literature)


class TestSearch:
    """Test cases for GraphExplorerService.search."""

    @pytest.mark.asyncio
    async def test_maps_hits_to_candidates(self):
        client = AsyncMock()
        client.search_entities.return_value = OpenTargetsResponse(
            success=True,
            data={"search": {"hits": [
                {"id": "ENSG00000138109", "entity": "target", "name": "CYP2C9",
                 "description": "cytochrome P450", "score": 4782.85},
                {"id": "CHEMBL1464", "entity": "drug", "name": "WARFARIN",
                 "description": None, "score": 100.0},
                {"id": "EFO_0000001", "entity": "disease", "name": "some disease",
                 "description": None, "score": 10.0},
            ]}},
        )
        service = make_service(client)

        result = await service.search("CYP2C9", limit=10)

        assert [c.id for c in result.candidates] == ["ENSG00000138109", "CHEMBL1464"]

    @pytest.mark.asyncio
    async def test_returns_empty_candidates_on_failure(self):
        client = AsyncMock()
        client.search_entities.return_value = OpenTargetsResponse(
            success=False, data=None, error_message="rate limited",
        )
        service = make_service(client)

        result = await service.search("nonsense", limit=10)

        assert result.query == "nonsense"
        assert result.candidates == []
