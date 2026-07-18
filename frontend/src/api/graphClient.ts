import type { GraphSearchResponse, GraphExpandResponse } from '../types';

// In dev, calls go to /api and Vite proxies them to the backend on :8000.
// In production the frontend is served from a static host, so point it at the
// deployed API by setting VITE_API_BASE (e.g. https://pharmgraph-api.onrender.com).
const API_BASE = (import.meta.env.VITE_API_BASE ?? '').replace(/\/$/, '');

export async function searchGraph(query: string, limit = 10): Promise<GraphSearchResponse> {
  const params = new URLSearchParams({ q: query, limit: String(limit) });
  const res = await fetch(`${API_BASE}/api/v1/graph/search?${params}`);
  if (!res.ok) {
    throw new Error(`Search failed: ${res.status}`);
  }
  return res.json();
}

export async function expandNode(
  nodeType: 'gene' | 'drug',
  nodeId: string,
  options: { limit?: number; minEvidence?: string | null } = {}
): Promise<GraphExpandResponse> {
  const { limit = 15, minEvidence } = options;
  const params = new URLSearchParams({ limit: String(limit) });
  if (minEvidence) {
    params.set('min_evidence', minEvidence);
  }
  const res = await fetch(
    `${API_BASE}/api/v1/graph/expand/${nodeType}/${encodeURIComponent(nodeId)}?${params}`
  );
  if (!res.ok) {
    throw new Error(`Expand failed: ${res.status}`);
  }
  return res.json();
}
