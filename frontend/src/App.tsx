import { useCallback, useEffect, useRef, useState } from 'react';
import { SearchBar } from './components/SearchBar';
import { GraphCanvas } from './components/GraphCanvas';
import { Legend } from './components/Legend';
import { NodeTooltip } from './components/NodeTooltip';
import { expandNode } from './api/graphClient';
import { CPIC_LEVELS } from './types';
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
  const [minEvidence, setMinEvidence] = useState<string>('');
  // The last root ("replace") expansion, so the evidence filter can re-run it.
  const lastEntryRef = useRef<{ nodeType: 'gene' | 'drug'; nodeId: string } | null>(null);

  const runExpand = useCallback(
    async (nodeType: 'gene' | 'drug', nodeId: string, mode: 'replace' | 'merge') => {
      setLoading(true);
      setError(null);
      try {
        const response = await expandNode(nodeType, nodeId, { minEvidence: minEvidence || null });
        if (response.nodes.length === 0) {
          setError(`No pharmacogenomic interactions found for ${nodeId}${minEvidence ? ` at evidence ${minEvidence} or stronger` : ''}.`);
        }
        setGraphState((prev) => mergeResponse(mode === 'replace' ? null : prev, response));
      } catch (err) {
        console.error(err);
        setError('Failed to load graph data. Is the backend running on :8000?');
      } finally {
        setLoading(false);
      }
    },
    [minEvidence]
  );

  const handleSelectCandidate = useCallback(
    (candidate: GraphSearchCandidate) => {
      const nodeType = toEntryType(candidate.entity_type);
      if (!nodeType) {
        setError(`Unsupported entity type: ${candidate.entity_type}`);
        return;
      }
      lastEntryRef.current = { nodeType, nodeId: candidate.id };
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

  // Re-run the current root expansion whenever the evidence filter changes.
  const didMountRef = useRef(false);
  useEffect(() => {
    if (!didMountRef.current) {
      didMountRef.current = true;
      return;
    }
    const entry = lastEntryRef.current;
    if (entry) {
      runExpand(entry.nodeType, entry.nodeId, 'replace');
    }
  }, [minEvidence, runExpand]);

  return (
    <div className="app">
      <header className="app-header">
        <h1>Gene · Protein · Drug Interaction Explorer</h1>
        <p className="app-subtitle">
          Live pharmacogenomic interaction data from Open Targets (PharmGKB / CPIC evidence)
        </p>
        <div className="controls">
          <SearchBar onSelect={handleSelectCandidate} />
          <label className="evidence-filter">
            <span>Min. evidence</span>
            <select value={minEvidence} onChange={(e) => setMinEvidence(e.target.value)}>
              <option value="">All levels</option>
              {CPIC_LEVELS.map((lvl) => (
                <option key={lvl} value={lvl}>
                  {lvl} or stronger
                </option>
              ))}
            </select>
          </label>
        </div>
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
