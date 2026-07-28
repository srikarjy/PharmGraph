export type GraphNodeType = 'gene' | 'protein' | 'drug';
export type GraphEdgeType = 'encodes' | 'pgx_interaction';

export interface GraphNode {
  id: string;
  type: GraphNodeType;
  label: string;
  subtitle?: string | null;
  metadata: Record<string, unknown>;
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  relationship: GraphEdgeType;
  action_type?: string | null;
  phenotype?: string | null;
  evidence_level?: string | null;
  annotation_count: number;
  confidence: number;
  literature: string[];
  pharmgkb_ids: string[];
}

export const CPIC_LEVELS = ['1A', '1B', '2A', '2B', '3', '4'] as const;

export interface GraphSearchCandidate {
  id: string;
  entity_type: string;
  name: string;
  description?: string | null;
  score: number;
}

export interface GraphSearchResponse {
  query: string;
  candidates: GraphSearchCandidate[];
}

export interface GraphExpandResponse {
  center_node_id: string;
  nodes: GraphNode[];
  edges: GraphEdge[];
  truncated: boolean;
  total_available?: number | null;
}

// Cell-cell communication network (ligand-receptor permutation test)
export interface CellCommNode {
  id: string;
  label: string;
  n_cells: number;
}

export interface CellCommEdge {
  id: string;
  source: string;
  target: string;
  ligand: string;
  receptor: string;
  ligand_mean_expression: number;
  receptor_mean_expression: number;
  interaction_score: number;
  p_value: number;
}

export interface CellCommDemoDataset {
  name: string;
  description: string;
  n_cells: number;
  n_genes: number;
  cell_types: string[];
}

export interface CellCommDemoDatasetsResponse {
  datasets: CellCommDemoDataset[];
}

export interface CellCommInferResponse {
  dataset_source: string;
  cell_type_key: string;
  n_cells: number;
  n_genes_tested: number;
  n_cell_types: number;
  n_permutations: number;
  pvalue_threshold: number;
  nodes: CellCommNode[];
  edges: CellCommEdge[];
  n_pairs_tested: number;
  n_significant: number;
}
