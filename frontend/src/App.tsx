import { useCallback, useState } from 'react';
import { SearchBar } from './components/SearchBar';
import { GraphCanvas } from './components/GraphCanvas';
import { Legend } from './components/Legend';
import { NodeTooltip } from './components/NodeTooltip';
import { expandNode } from './api/graphClient';
import type { GraphSearchCandidate, GraphExpandResponse, GraphNode, GraphEdge } from './types';
import './App.css';

interface GraphState {
  nodes: GraphNode[];
  links: GraphEdge[];
}

function toEntryType(entityType: string): 'gene' | 'drug' | null {
  if (entityType === 'target') return 'gene';
  if (entityType === 'drug') return 'drug';
  return null;
}

// Merges a fresh subgraph into existing state without replacing node/link object
// references that are already known -- react-force-graph mutates node objects in
// place (x/y/vx/vy) once rendered, so replacing the whole array on every expand
// would reset the simulation and jump the layout on every click.
function mergeResponse(prev: GraphState | null, response: GraphExpandResponse): GraphState {
  const existingNodeIds = new Set(prev?.nodes.map((n) => n.id) ?? []);
  const existingEdgeIds = new Set(prev?.links.map((l) => l.id) ?? []);

  const newNodes = response.nodes.filter((n) => !existingNodeIds.has(n.id));
  const newLinks = response.edges.filter((e) => !existingEdgeIds.has(e.id));

  return {
    nodes: [...(prev?.nodes ?? []), ...newNodes],
    links: [...(prev?.links ?? []), ...newLinks],
  };
}

export default function App() {
  const [graphState, setGraphState] = useState<GraphState | null>(null);
  const [hovered, setHovered] = useState<GraphNode | GraphEdge | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const runExpand = useCallback(
    async (nodeType: 'gene' | 'drug', nodeId: string, mode: 'replace' | 'merge') => {
      setLoading(true);
      setError(null);
      try {
        const response = await expandNode(nodeType, nodeId);
        if (response.nodes.length === 0) {
          setError(`No pharmacogenomic interactions found for ${nodeId}.`);
        }
        setGraphState((prev) => mergeResponse(mode === 'replace' ? null : prev, response));
      } catch (err) {
        console.error(err);
        setError('Failed to load graph data. Is the backend running on :8000?');
      } finally {
        setLoading(false);
      }
    },
    []
  );

  const handleSelectCandidate = useCallback(
    (candidate: GraphSearchCandidate) => {
      const nodeType = toEntryType(candidate.entity_type);
      if (!nodeType) {
        setError(`Unsupported entity type: ${candidate.entity_type}`);
        return;
      }
      runExpand(nodeType, candidate.id, 'replace');
    },
    [runExpand]
  );

  const handleNodeClick = useCallback(
    (node: GraphNode) => {
      if (node.type === 'protein') return;
      runExpand(node.type as 'gene' | 'drug', node.id, 'merge');
    },
    [runExpand]
  );

  return (
    <div className="app">
      <header className="app-header">
        <h1>Gene · Protein · Drug Interaction Explorer</h1>
        <p className="app-subtitle">
          Live pharmacogenomic interaction data from Open Targets (PharmGKB / CPIC evidence)
        </p>
        <SearchBar onSelect={handleSelectCandidate} />
      </header>

      <main className="app-main">
        <GraphCanvas
          data={graphState}
          onNodeClick={handleNodeClick}
          onHoverChange={setHovered}
        />
        <aside className="side-panel">
          <Legend />
          <NodeTooltip item={hovered} />
        </aside>
      </main>

      {loading && <div className="status-banner status-loading">Loading…</div>}
      {error && <div className="status-banner status-error">{error}</div>}
    </div>
  );
}
