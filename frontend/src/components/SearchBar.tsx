import { useEffect, useRef, useState } from 'react';
import { searchGraph } from '../api/graphClient';
import type { GraphSearchCandidate } from '../types';

interface SearchBarProps {
  onSelect: (candidate: GraphSearchCandidate) => void;
}

export function SearchBar({ onSelect }: SearchBarProps) {
  const [query, setQuery] = useState('');
  const [candidates, setCandidates] = useState<GraphSearchCandidate[]>([]);
  const [loading, setLoading] = useState(false);
  const [open, setOpen] = useState(false);
  const debounceRef = useRef<number | undefined>(undefined);

  useEffect(() => {
    if (!query.trim()) {
      setCandidates([]);
      setOpen(false);
      return;
    }

    window.clearTimeout(debounceRef.current);
    debounceRef.current = window.setTimeout(async () => {
      setLoading(true);
      try {
        const result = await searchGraph(query);
        setCandidates(result.candidates);
        setOpen(true);
      } catch (err) {
        console.error(err);
        setCandidates([]);
      } finally {
        setLoading(false);
      }
    }, 300);

    return () => window.clearTimeout(debounceRef.current);
  }, [query]);

  return (
    <div className="search-bar">
      <input
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        onFocus={() => candidates.length > 0 && setOpen(true)}
        onBlur={() => window.setTimeout(() => setOpen(false), 150)}
        placeholder="Search a gene, protein, or drug (e.g. CYP2C9, warfarin)"
      />
      {loading && <div className="search-status">Searching…</div>}
      {open && candidates.length > 0 && (
        <ul className="search-dropdown">
          {candidates.map((c) => (
            <li
              key={c.id}
              onMouseDown={() => {
                onSelect(c);
                setQuery(c.name);
                setOpen(false);
              }}
            >
              <span className={`badge badge-${c.entity_type}`}>{c.entity_type}</span>
              <span className="candidate-name">{c.name}</span>
              {c.description && <span className="candidate-desc">{c.description}</span>}
            </li>
          ))}
        </ul>
      )}
      {open && !loading && candidates.length === 0 && query.trim() && (
        <div className="search-dropdown search-empty">No matches found.</div>
      )}
    </div>
  );
}
