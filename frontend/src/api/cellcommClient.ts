import type { CellCommDemoDatasetsResponse, CellCommInferResponse } from '../types';

// Same dev-proxy / VITE_API_BASE convention as graphClient.ts.
const API_BASE = (import.meta.env.VITE_API_BASE ?? '').replace(/\/$/, '');

export async function listCellCommDemoDatasets(): Promise<CellCommDemoDatasetsResponse> {
  const res = await fetch(`${API_BASE}/api/v1/cellcomm/demo-datasets`);
  if (!res.ok) {
    throw new Error(`Failed to list demo datasets: ${res.status}`);
  }
  return res.json();
}

export interface InferCellCommOptions {
  cellTypeKey?: string;
  nPermutations?: number;
  pvalueThreshold?: number;
  minCellsPerType?: number;
}

export async function inferCellComm(
  dataset: string,
  options: InferCellCommOptions = {}
): Promise<CellCommInferResponse> {
  const {
    cellTypeKey = 'cell_type',
    nPermutations = 200,
    pvalueThreshold = 0.05,
    minCellsPerType = 10,
  } = options;

  const params = new URLSearchParams({
    dataset,
    cell_type_key: cellTypeKey,
    n_permutations: String(nPermutations),
    pvalue_threshold: String(pvalueThreshold),
    min_cells_per_type: String(minCellsPerType),
  });

  const res = await fetch(`${API_BASE}/api/v1/cellcomm/infer?${params}`, { method: 'POST' });
  if (!res.ok) {
    throw new Error(`Inference failed: ${res.status}`);
  }
  return res.json();
}
