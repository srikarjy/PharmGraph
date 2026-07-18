import { useEffect, useRef, useState } from 'react';
import ForceGraph2D, { type ForceGraphMethods } from 'react-force-graph-2d';
import type { GraphNode, GraphEdge } from '../types';

const NODE_COLORS: Record<string, string> = {
  gene: '#4c6ef5',
  protein: '#845ef7',
  drug: '#2f9e44',
};

interface GraphCanvasProps {
  data: { nodes: GraphNode[]; links: GraphEdge[] } | null;
  onNodeClick: (node: GraphNode) => void;
  onHoverChange: (item: GraphNode | GraphEdge | null) => void;
}

export function GraphCanvas({ data, onNodeClick, onHoverChange }: GraphCanvasProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const fgRef = useRef<ForceGraphMethods<GraphNode, GraphEdge> | undefined>(undefined);
  const [size, setSize] = useState({ width: 800, height: 600 });

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (entry) {
        setSize({ width: entry.contentRect.width, height: entry.contentRect.height });
      }
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  if (!data || data.nodes.length === 0) {
    return (
      <div ref={containerRef} className="graph-canvas graph-placeholder">
        Search for a gene, protein, or drug above to begin exploring interactions.
      </div>
    );
  }

  return (
    <div ref={containerRef} className="graph-canvas">
      <ForceGraph2D<GraphNode, GraphEdge>
        ref={fgRef}
        width={size.width}
        height={size.height}
        graphData={data}
        nodeId="id"
        nodeLabel={(n) => `${n.label}${n.subtitle ? ` — ${n.subtitle}` : ''}`}
        nodeColor={(n) => NODE_COLORS[n.type] ?? '#868e96'}
        nodeRelSize={5}
        linkColor={() => 'rgba(148, 163, 184, 0.55)'}
        linkWidth={(l) => 1 + (l.confidence ?? 0.3) * 3}
        linkDirectionalParticles={(l) => (l.relationship === 'pgx_interaction' ? 2 : 0)}
        linkDirectionalParticleWidth={1.5}
        linkLabel={(l) =>
          l.relationship === 'pgx_interaction'
            ? `${l.action_type ?? 'interaction'}: ${l.phenotype ?? ''} (evidence ${l.evidence_level ?? 'n/a'})`
            : 'encodes'
        }
        onNodeClick={(node) => onNodeClick(node)}
        onNodeHover={(node) => onHoverChange(node)}
        onLinkHover={(link) => onHoverChange(link)}
        cooldownTicks={100}
      />
    </div>
  );
}
