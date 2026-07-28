import { useEffect, useState } from 'react';
import { listCellCommDemoDatasets } from '../api/cellcommClient';
import type { CellCommDemoDataset } from '../types';

interface CellCommControlsProps {
  onRun: (dataset: string, nPermutations: number, pvalueThreshold: number) => void;
  loading: boolean;
}

export function CellCommControls({ onRun, loading }: CellCommControlsProps) {
  const [datasets, setDatasets] = useState<CellCommDemoDataset[]>([]);
  const [dataset, setDataset] = useState<string>('');
  const [nPermutations, setNPermutations] = useState(200);
  const [pvalueThreshold, setPvalueThreshold] = useState(0.05);

  useEffect(() => {
    listCellCommDemoDatasets()
      .then((res) => {
        setDatasets(res.datasets);
        if (res.datasets.length > 0) setDataset(res.datasets[0].name);
      })
      .catch((err) => console.error(err));
  }, []);

  const selected = datasets.find((d) => d.name === dataset);

  return (
    <div className="cellcomm-controls">
      <div className="controls">
        <label className="evidence-filter">
          <span>Dataset</span>
          <select value={dataset} onChange={(e) => setDataset(e.target.value)}>
            {datasets.map((d) => (
              <option key={d.name} value={d.name}>
                {d.name} ({d.n_cells} cells, {d.cell_types.length} cell types)
              </option>
            ))}
          </select>
        </label>

        <label className="evidence-filter">
          <span>Permutations: {nPermutations}</span>
          <input
            type="range"
            min={50}
            max={1000}
            step={50}
            value={nPermutations}
            onChange={(e) => setNPermutations(Number(e.target.value))}
          />
        </label>

        <label className="evidence-filter">
          <span>Significance threshold: {pvalueThreshold.toFixed(2)}</span>
          <input
            type="range"
            min={0.01}
            max={0.2}
            step={0.01}
            value={pvalueThreshold}
            onChange={(e) => setPvalueThreshold(Number(e.target.value))}
          />
        </label>

        <button
          className="run-button"
          disabled={!dataset || loading}
          onClick={() => onRun(dataset, nPermutations, pvalueThreshold)}
        >
          {loading ? 'Running…' : 'Run inference'}
        </button>
      </div>
      {selected && <p className="cellcomm-dataset-desc">{selected.description}</p>}
    </div>
  );
}
