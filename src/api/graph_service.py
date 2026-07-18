"""Service layer for the gene/protein/drug interaction graph explorer.

Aggregates Open Targets' raw pharmacogenomics rows (hundreds of duplicated
rows per gene/drug, one per allele/genotype/study) into a clean graph of
deduplicated nodes and edges, keeping the highest-evidence annotation per
drug-gene pair. See docs/gene_drug_protein_graph_blueprint.md for the schema
this logic is built against.
"""

from typing import Dict, Any, List, Optional

from ..config import get_logger
from ..data_ingestion.opentargets_client import OpenTargetsClient
from .schemas import (
    GraphNode, GraphEdge, GraphNodeType, GraphEdgeType,
    GraphSearchCandidate, GraphSearchResponse, GraphExpandResponse,
)

logger = get_logger(__name__)

# CPIC evidence level -> confidence score, and rank for picking the best annotation per group.
EVIDENCE_CONFIDENCE = {"1A": 0.95, "1B": 0.85, "2A": 0.70, "2B": 0.60, "3": 0.40, "4": 0.25}
DEFAULT_CONFIDENCE = 0.30
EVIDENCE_RANK = {"1A": 0, "1B": 1, "2A": 2, "2B": 3, "3": 4, "4": 5}


def _evidence_rank(level: Optional[str]) -> int:
    return EVIDENCE_RANK.get(level, len(EVIDENCE_RANK))


def _confidence_for(level: Optional[str]) -> float:
    return EVIDENCE_CONFIDENCE.get(level, DEFAULT_CONFIDENCE)


def _meets_min_evidence(level: Optional[str], min_level: Optional[str]) -> bool:
    """Whether `level` is at least as strong as `min_level` on the CPIC scale.

    Lower rank = stronger evidence (1A is rank 0). An unknown/missing level ranks
    below every named tier, so it is excluded whenever a threshold is set.
    """
    if not min_level:
        return True
    return _evidence_rank(level) <= _evidence_rank(min_level)


# Cap PubMed IDs surfaced per edge so a heavily-studied pair doesn't bloat the payload.
MAX_LITERATURE_PER_EDGE = 12

# Cap PharmGKB clinical-annotation links per edge for the same reason.
MAX_PHARMGKB_PER_EDGE = 12

# Open Targets sources all pharmacogenomics rows from PharmGKB (now branded ClinPGx);
# its studyId is the PharmGKB clinical-annotation accession.
PHARMGKB_DATASOURCE = "clinpgx"


def _pharmgkb_id(row: Dict[str, Any]) -> Optional[str]:
    """Return the PharmGKB clinical-annotation id for a pgx row, if it has one."""
    if row.get("datasourceId") == PHARMGKB_DATASOURCE:
        return row.get("studyId")
    return None


class GraphExplorerService:
    """Aggregates raw Open Targets pharmacogenomics data into a clean interaction graph."""

    def __init__(self, client: OpenTargetsClient):
        """Initialize the service.

        Args:
            client: An entered (session-active) OpenTargetsClient
        """
        self.client = client

    async def search(self, query: str, limit: int) -> GraphSearchResponse:
        """Search for genes/proteins/drugs by name.

        Args:
            query: Free-text search string
            limit: Maximum number of candidates to return

        Returns:
            GraphSearchResponse: Matching entities, or empty on failure
        """
        try:
            result = await self.client.search_entities(query, limit=limit)
            if not result.success or not result.data:
                logger.warning(f"Open Targets search failed for '{query}': {result.error_message}")
                return GraphSearchResponse(query=query, candidates=[])

            hits = result.data.get("search", {}).get("hits", [])
            candidates = [
                GraphSearchCandidate(
                    id=hit["id"],
                    entity_type=hit["entity"],
                    name=hit["name"],
                    description=hit.get("description"),
                    score=hit.get("score", 0.0),
                )
                for hit in hits
                if hit.get("entity") in ("target", "drug")
            ]
            return GraphSearchResponse(query=query, candidates=candidates)

        except Exception as e:
            logger.error(f"Graph search failed for '{query}': {e}")
            return GraphSearchResponse(query=query, candidates=[])

    async def expand(
        self, node_type: str, node_id: str, limit: int, min_evidence: Optional[str] = None
    ) -> GraphExpandResponse:
        """Expand a gene or drug node into its pharmacogenomic interaction subgraph.

        Args:
            node_type: 'gene' or 'drug'
            node_id: Ensembl gene ID (for gene) or ChEMBL ID (for drug)
            limit: Maximum number of interaction partners to include
            min_evidence: If set (e.g. '2A'), drop interactions weaker than this CPIC tier

        Returns:
            GraphExpandResponse: Subgraph nodes/edges, empty on failure
        """
        try:
            # "target" is Open Targets' term for a gene and is what search() emits,
            # so accept it as an alias to keep search -> expand directly composable.
            if node_type in ("gene", "target"):
                return await self._expand_gene(node_id, limit, min_evidence)
            if node_type == "drug":
                return await self._expand_drug(node_id, limit, min_evidence)
            raise ValueError(f"Unsupported node_type: {node_type}")

        except Exception as e:
            logger.error(f"Graph expand failed for {node_type}/{node_id}: {e}")
            return GraphExpandResponse(center_node_id=node_id, nodes=[], edges=[])

    async def _expand_gene(
        self, ensembl_id: str, limit: int, min_evidence: Optional[str] = None
    ) -> GraphExpandResponse:
        result = await self.client.get_target_with_pharmacogenomics(ensembl_id)
        if not result.success or not result.data or not result.data.get("target"):
            return GraphExpandResponse(center_node_id=ensembl_id, nodes=[], edges=[])

        target = result.data["target"]
        nodes: List[GraphNode] = []
        edges: List[GraphEdge] = []

        gene_node = GraphNode(
            id=target["id"], type=GraphNodeType.GENE, label=target["approvedSymbol"],
            subtitle=target.get("approvedName"),
        )
        nodes.append(gene_node)

        protein_id = self._canonical_protein_id(target.get("proteinIds") or [])
        if protein_id:
            nodes.append(GraphNode(id=protein_id, type=GraphNodeType.PROTEIN, label=protein_id, subtitle="UniProt"))
            edges.append(GraphEdge(
                id=f"{gene_node.id}:{protein_id}:encodes",
                source=gene_node.id, target=protein_id, relationship=GraphEdgeType.ENCODES,
            ))

        groups = [
            g for g in self._group_by_drug(target.get("pharmacogenomics") or []).values()
            if _meets_min_evidence(g["evidence_level"], min_evidence)
        ]
        total_available = len(groups)
        top_groups = sorted(
            groups, key=lambda g: _confidence_for(g["evidence_level"]), reverse=True
        )[:limit]

        for group in top_groups:
            nodes.append(GraphNode(id=group["drug_id"], type=GraphNodeType.DRUG, label=group["drug_name"]))
            edges.append(GraphEdge(
                id=f"{gene_node.id}:{group['drug_id']}:pgx_interaction",
                source=gene_node.id, target=group["drug_id"], relationship=GraphEdgeType.PGX_INTERACTION,
                action_type=group["pgx_category"], phenotype=group["phenotype_text"],
                evidence_level=group["evidence_level"], annotation_count=group["count"],
                confidence=_confidence_for(group["evidence_level"]),
                literature=sorted(group["literature"])[:MAX_LITERATURE_PER_EDGE],
                pharmgkb_ids=sorted(group["pharmgkb"])[:MAX_PHARMGKB_PER_EDGE],
            ))

        return GraphExpandResponse(
            center_node_id=ensembl_id, nodes=nodes, edges=edges,
            truncated=total_available > limit, total_available=total_available,
        )

    async def _expand_drug(
        self, chembl_id: str, limit: int, min_evidence: Optional[str] = None
    ) -> GraphExpandResponse:
        result = await self.client.get_drug_with_pharmacogenomics(chembl_id)
        if not result.success or not result.data or not result.data.get("drug"):
            return GraphExpandResponse(center_node_id=chembl_id, nodes=[], edges=[])

        drug = result.data["drug"]
        nodes: List[GraphNode] = []
        edges: List[GraphEdge] = []

        drug_node = GraphNode(id=drug["id"], type=GraphNodeType.DRUG, label=drug["name"])
        nodes.append(drug_node)

        groups = [
            g for g in self._group_by_gene(drug.get("pharmacogenomics") or []).values()
            if _meets_min_evidence(g["evidence_level"], min_evidence)
        ]
        total_available = len(groups)
        top_groups = sorted(
            groups, key=lambda g: _confidence_for(g["evidence_level"]), reverse=True
        )[:limit]

        for group in top_groups:
            nodes.append(GraphNode(id=group["gene_id"], type=GraphNodeType.GENE, label=group["gene_symbol"]))
            edges.append(GraphEdge(
                id=f"{group['gene_id']}:{drug_node.id}:pgx_interaction",
                source=group["gene_id"], target=drug_node.id, relationship=GraphEdgeType.PGX_INTERACTION,
                action_type=group["pgx_category"], phenotype=group["phenotype_text"],
                evidence_level=group["evidence_level"], annotation_count=group["count"],
                confidence=_confidence_for(group["evidence_level"]),
                literature=sorted(group["literature"])[:MAX_LITERATURE_PER_EDGE],
                pharmgkb_ids=sorted(group["pharmgkb"])[:MAX_PHARMGKB_PER_EDGE],
            ))

        return GraphExpandResponse(
            center_node_id=chembl_id, nodes=nodes, edges=edges,
            truncated=total_available > limit, total_available=total_available,
        )

    @staticmethod
    def _canonical_protein_id(protein_ids: List[Dict[str, str]]) -> Optional[str]:
        """Pick the canonical UniProt Swiss-Prot id, skipping TrEMBL/obsolete/Ensembl-PRO entries."""
        for entry in protein_ids:
            if entry.get("source") == "uniprot_swissprot":
                return entry.get("id")
        return None

    @staticmethod
    def _group_by_drug(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Group gene-side pharmacogenomics rows by drug, keeping the best evidence per drug."""
        groups: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            for drug_entry in row.get("drugs") or []:
                drug_id = drug_entry.get("drugId")
                if not drug_id:
                    continue
                drug_name = (drug_entry.get("drug") or {}).get("name") or drug_id

                pgkb = _pharmgkb_id(row)
                existing = groups.get(drug_id)
                if existing is None:
                    groups[drug_id] = {
                        "drug_id": drug_id, "drug_name": drug_name,
                        "pgx_category": row.get("pgxCategory"),
                        "phenotype_text": row.get("phenotypeText"),
                        "evidence_level": row.get("evidenceLevel"),
                        "literature": set(row.get("literature") or []),
                        "pharmgkb": {pgkb} if pgkb else set(),
                        "count": 1,
                    }
                else:
                    existing["count"] += 1
                    existing["literature"].update(row.get("literature") or [])
                    if pgkb:
                        existing["pharmgkb"].add(pgkb)
                    if _evidence_rank(row.get("evidenceLevel")) < _evidence_rank(existing["evidence_level"]):
                        existing["pgx_category"] = row.get("pgxCategory")
                        existing["phenotype_text"] = row.get("phenotypeText")
                        existing["evidence_level"] = row.get("evidenceLevel")
        return groups

    @staticmethod
    def _group_by_gene(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Group drug-side pharmacogenomics rows by gene, keeping the best evidence per gene."""
        groups: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            target = row.get("target")
            if not target or not target.get("id"):
                continue
            gene_id = target["id"]
            gene_symbol = target.get("approvedSymbol") or gene_id

            pgkb = _pharmgkb_id(row)
            existing = groups.get(gene_id)
            if existing is None:
                groups[gene_id] = {
                    "gene_id": gene_id, "gene_symbol": gene_symbol,
                    "pgx_category": row.get("pgxCategory"),
                    "phenotype_text": row.get("phenotypeText"),
                    "evidence_level": row.get("evidenceLevel"),
                    "literature": set(row.get("literature") or []),
                    "pharmgkb": {pgkb} if pgkb else set(),
                    "count": 1,
                }
            else:
                existing["count"] += 1
                existing["literature"].update(row.get("literature") or [])
                if pgkb:
                    existing["pharmgkb"].add(pgkb)
                if _evidence_rank(row.get("evidenceLevel")) < _evidence_rank(existing["evidence_level"]):
                    existing["pgx_category"] = row.get("pgxCategory")
                    existing["phenotype_text"] = row.get("phenotypeText")
                    existing["evidence_level"] = row.get("evidenceLevel")
        return groups
