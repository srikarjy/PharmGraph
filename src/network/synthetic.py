"""Deterministic synthetic demo dataset for the cell-cell communication endpoint.

Not real patient data. A small, biologically-plausible AnnData with four
immune-like cell types and a handful of "planted" ligand-receptor signals on
top of background noise, so the demo reliably surfaces real interactions
without any network fetch or extra dependency (just numpy/pandas/anndata,
already required elsewhere).
"""

import numpy as np
import pandas as pd

from .lr_pairs import LR_PAIRS

try:
    import anndata as ad
except ImportError:  # pragma: no cover - exercised only when anndata is absent
    ad = None

CELL_TYPES = ["T_cell", "B_cell", "NK_cell", "Monocyte"]
N_CELLS_PER_TYPE = 100

# (gene, cell_type, target_mean) elevations planted on top of background noise so
# a few real, well-known ligand-receptor interactions are statistically
# recoverable by the demo:
#   T_cell   --CD40LG--> B_cell    (CD40)   classic T-B costimulation
#   Monocyte --IL6-----> T_cell    (IL6R)   myeloid-derived cytokine signaling
#   NK_cell  --CCL5----> Monocyte  (CCR5)   NK-derived chemokine recruitment
PLANTED_ELEVATIONS = [
    ("CD40LG", "T_cell", 9.0),
    ("CD40", "B_cell", 9.0),
    ("IL6", "Monocyte", 9.0),
    ("IL6R", "T_cell", 9.0),
    ("CCL5", "NK_cell", 9.0),
    ("CCR5", "Monocyte", 9.0),
]


def generate_demo_dataset(seed: int = 0) -> "ad.AnnData":
    """Build the synthetic demo AnnData, deterministic given `seed`."""
    if ad is None:
        raise ImportError("anndata is required to generate the demo dataset")

    rng = np.random.default_rng(seed)

    genes = sorted({g for pair in LR_PAIRS for g in (pair.ligand, pair.receptor)})
    gene_index = {g: i for i, g in enumerate(genes)}
    n_genes = len(genes)

    labels = np.repeat(CELL_TYPES, N_CELLS_PER_TYPE)
    n_cells = len(labels)

    # Background: low, noisy, non-negative expression on every gene/cell.
    x = rng.gamma(shape=2.0, scale=0.15, size=(n_cells, n_genes)).astype(np.float32)

    # Planted elevations: add a strong, noisy signal for specific gene/cell-type pairs.
    for gene, cell_type, target_mean in PLANTED_ELEVATIONS:
        gi = gene_index[gene]
        mask = labels == cell_type
        n = int(mask.sum())
        x[mask, gi] += rng.gamma(shape=4.0, scale=target_mean / 4.0, size=n).astype(np.float32)

    obs = pd.DataFrame({"cell_type": pd.Categorical(labels)})
    obs.index = [f"cell_{i}" for i in range(n_cells)]
    var = pd.DataFrame(index=genes)

    return ad.AnnData(X=x, obs=obs, var=var)
