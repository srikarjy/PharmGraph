import { useEffect, useRef, useState } from 'react';
import ForceGraph2D, { type ForceGraphMethods } from 'react-force-graph-2d';
import type { CellCommNode, CellCommEdge } from '../types';

interface CellCommCanvasProps {
  data: { nodes: CellCommNode[]; links: CellCommEdge[] } | null;
  onHoverChange: (item: CellCommNode | CellCommEdge | null) => void;
}

export function CellCommCanvas({ data, onHoverChange }: CellCommCanvasProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const fgRef = useRef<ForceGraphMethods<CellCommNode, CellCommEdge> | undefined>(undefined);
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
        Run inference below to explore the ligand-receptor network between cell types.
      </div>
    );
  }

  return (
    <div ref={containerRef} className="graph-canvas">
      <ForceGraph2D<CellCommNode, CellCommEdge>
        ref={fgRef}
        width={size.width}
        height={size.height}
        graphData={data}
        nodeId="id"
        nodeLabel={(n) => `${n.label} (${n.n_cells} cells)`}
        nodeColor={() => '#e8590c'}
        nodeVal={(n) => Math.max(2, Math.sqrt(n.n_cells))}
        linkColor={(l) => `rgba(232, 89, 12, ${Math.min(1, 0.25 + l.interaction_score / 10)})`}
        linkWidth={(l) => 1 + Math.min(4, l.interaction_score / 3)}
        linkDirectionalArrowLength={5}
        linkDirectionalArrowRelPos={1}
        linkDirectionalParticles={2}
        linkDirectionalParticleWidth={1.5}
        linkLabel={(l) => `${l.ligand} → ${l.receptor} (p=${l.p_value.toFixed(4)})`}
        onNodeHover={(node) => onHoverChange(node)}
        onLinkHover={(link) => onHoverChange(link)}
        cooldownTicks={100}
      />
    </div>
  );
}
