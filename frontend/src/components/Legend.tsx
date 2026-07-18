export function Legend() {
  return (
    <div className="legend">
      <div className="legend-item">
        <span className="dot dot-gene" />
        Gene
      </div>
      <div className="legend-item">
        <span className="dot dot-protein" />
        Protein
      </div>
      <div className="legend-item">
        <span className="dot dot-drug" />
        Drug
      </div>
      <p className="legend-hint">Click a gene or drug node to expand its interactions.</p>
    </div>
  );
}
