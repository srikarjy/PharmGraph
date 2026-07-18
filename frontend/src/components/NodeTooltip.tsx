import type { GraphNode, GraphEdge } from '../types';

function isEdge(item: GraphNode | GraphEdge): item is GraphEdge {
  return 'source' in item && 'target' in item;
}

interface NodeTooltipProps {
  item: GraphNode | GraphEdge | null;
}

export function NodeTooltip({ item }: NodeTooltipProps) {
  if (!item) {
    return <div className="detail-panel detail-panel-empty">Hover a node or edge to see details.</div>;
  }

  if (isEdge(item)) {
    if (item.relationship !== 'pgx_interaction') {
      return (
        <div className="detail-panel">
          <h3>Encodes</h3>
          <p>Gene encodes this protein.</p>
        </div>
      );
    }

    return (
      <div className="detail-panel">
        <h3>Pharmacogenomic interaction</h3>
        <dl>
          <dt>Category</dt>
          <dd>{item.action_type ?? 'unknown'}</dd>
          <dt>Phenotype</dt>
          <dd>{item.phenotype ?? 'n/a'}</dd>
          <dt>Evidence level</dt>
          <dd>{item.evidence_level ?? 'n/a'} (CPIC)</dd>
          <dt>Confidence</dt>
          <dd>{Math.round(item.confidence * 100)}%</dd>
          <dt>Source annotations</dt>
          <dd>{item.annotation_count}</dd>
        </dl>
      </div>
    );
  }

  return (
    <div className="detail-panel">
      <h3>{item.label}</h3>
      <p className="detail-type">{item.type}</p>
      {item.subtitle && <p className="detail-subtitle">{item.subtitle}</p>}
      {item.type !== 'protein' && <p className="detail-hint">Click to expand its network</p>}
    </div>
  );
}
