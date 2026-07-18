import type { GraphSearchResponse, GraphExpandResponse } from '../types';

export async function searchGraph(query: string, limit = 10): Promise<GraphSearchResponse> {
  const params = new URLSearchParams({ q: query, limit: String(limit) });
  const res = await fetch(`/api/v1/graph/search?${params}`);
  if (!res.ok) {
    throw new Error(`Search failed: ${res.status}`);
  }
  return res.json();
}

export async function expandNode(
  nodeType: 'gene' | 'drug',
  nodeId: string,
  limit = 15
): Promise<GraphExpandResponse> {
  const params = new URLSearchParams({ limit: String(limit) });
  const res = await fetch(`/api/v1/graph/expand/${nodeType}/${encodeURIComponent(nodeId)}?${params}`);
  if (!res.ok) {
    throw new Error(`Expand failed: ${res.status}`);
  }
  return res.json();
}
