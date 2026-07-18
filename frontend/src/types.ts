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
