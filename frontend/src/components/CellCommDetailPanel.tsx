import type { CellCommNode, CellCommEdge, CellCommInferResponse } from '../types';

function isEdge(item: CellCommNode | CellCommEdge): item is CellCommEdge {
  return 'source' in item && 'target' in item;
}

interface CellCommDetailPanelProps {
  item: CellCommNode | CellCommEdge | null;
  result: CellCommInferResponse | null;
}

export function CellCommDetailPanel({ item, result }: CellCommDetailPanelProps) {
  if (!item) {
    if (!result) {
      return <div className="detail-panel detail-panel-empty">Run inference to see results.</div>;
    }
    return (
      <div className="detail-panel">
        <h3>Run summary</h3>
        <dl>
          <dt>Cells analyzed</dt>
          <dd>{result.n_cells}</dd>
          <dt>Ligand/receptor genes tested</dt>
          <dd>{result.n_genes_tested}</dd>
          <dt>Pairs tested</dt>
          <dd>{result.n_pairs_tested}</dd>
          <dt>Permutations</dt>
          <dd>{result.n_permutations}</dd>
          <dt>Significance threshold</dt>
          <dd>{result.pvalue_threshold}</dd>
          <dt>Significant interactions</dt>
          <dd>{result.n_significant}</dd>
        </dl>
        <p className="detail-hint">Hover a node or edge to see details.</p>
      </div>
    );
  }

  if (isEdge(item)) {
    return (
      <div className="detail-panel">
        <h3>Ligand-receptor interaction</h3>
        <dl>
          <dt>Ligand</dt>
          <dd>{item.ligand}</dd>
          <dt>Receptor</dt>
          <dd>{item.receptor}</dd>
          <dt>Sending → receiving</dt>
          <dd>{item.source} → {item.target}</dd>
          <dt>Ligand mean expression</dt>
          <dd>{item.ligand_mean_expression.toFixed(2)}</dd>
          <dt>Receptor mean expression</dt>
          <dd>{item.receptor_mean_expression.toFixed(2)}</dd>
          <dt>Interaction score</dt>
          <dd>{item.interaction_score.toFixed(2)}</dd>
          <dt>p-value</dt>
          <dd>{item.p_value.toFixed(4)}</dd>
        </dl>
      </div>
    );
  }

  return (
    <div className="detail-panel">
      <h3>{item.label}</h3>
      <p className="detail-type">cell type</p>
      <dl>
        <dt>Cells</dt>
        <dd>{item.n_cells}</dd>
      </dl>
    </div>
  );
}
