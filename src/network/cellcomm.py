"""Cell-cell communication network inference: a CellPhoneDB-style permutation test.

Pure numpy/anndata -- no FastAPI or pydantic imports -- so the algorithm is
directly unit-testable and reusable outside the API layer. See
docs/cellcomm_blueprint.md for the full algorithm writeup.

Given a single-cell dataset (AnnData: cells x genes, with a categorical
cell-type column) and a set of ligand-receptor gene pairs, this infers which
(ligand, receptor, sending cell type, receiving cell type) interactions have
significantly elevated expression relative to a null distribution built by
permuting cell-type labels.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

from . import caps

try:
    import anndata as ad
except ImportError:  # pragma: no cover - exercised only when anndata is absent
    ad = None


@dataclass
class InteractionResult:
    """A single significant ligand-receptor interaction between two cell types."""
    ligand: str
    receptor: str
    source_cluster: str
    target_cluster: str
    ligand_mean_expression: float
    receptor_mean_expression: float
    interaction_score: float
    p_value: float


@dataclass
class CellCommResult:
    """The outcome of one `infer_cell_communication` call."""
    cell_types: List[str]
    cell_counts: List[int]
    n_cells: int
    n_genes_tested: int
    n_permutations: int
    pvalue_threshold: float
    n_pairs_tested: int
    interactions: List[InteractionResult] = field(default_factory=list)

    @property
    def n_significant(self) -> int:
        return len(self.interactions)


def _one_hot(labels: np.ndarray, categories: Sequence[str]) -> np.ndarray:
    """Vectorized one-hot encode of `labels` against a fixed category order."""
    categories = np.asarray(categories)
    return (labels[:, None] == categories[None, :]).astype(np.float64)


def _cluster_means(X: np.ndarray, labels: np.ndarray, categories: Sequence[str]) -> np.ndarray:
    """Per-cluster mean expression via one indicator-matrix matmul.

    Returns a (K, n_genes) array, row k = mean expression profile of
    categories[k]. This single BLAS call replaces a Python loop over clusters
    (`for c in categories: X[labels == c].mean(axis=0)`) -- it's the operation
    re-run on every permutation, so its cost dominates the whole test.
    """
    indicator = _one_hot(labels, categories)          # (n_cells, K)
    sums = indicator.T @ X                             # (K, n_genes)
    counts = indicator.sum(axis=0)                     # (K,)
    return sums / counts[:, None]


def _lr_scores(
    cluster_means: np.ndarray, ligand_idx: np.ndarray, receptor_idx: np.ndarray
) -> np.ndarray:
    """Interaction score for every (pair, source_cluster, target_cluster) triple.

    score[p, A, B] = (mean(ligand_p, A) + mean(receptor_p, B)) / 2

    Computed for both orderings of every cluster pair (including A == B,
    autocrine signaling) since ligand/receptor roles are asymmetric. Returns
    shape (n_pairs, K, K), fully vectorized via broadcasting.
    """
    mean_l = cluster_means[:, ligand_idx]      # (K, n_pairs)
    mean_r = cluster_means[:, receptor_idx]    # (K, n_pairs)
    return (mean_l.T[:, :, None] + mean_r.T[:, None, :]) / 2.0


def _permutation_pvalues(
    X: np.ndarray,
    labels: np.ndarray,
    categories: Sequence[str],
    ligand_idx: np.ndarray,
    receptor_idx: np.ndarray,
    observed_scores: np.ndarray,
    n_permutations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """One-sided empirical p-values from label-permutation null scores.

    Shuffling `labels` preserves cluster sizes but destroys any real
    association between a cell's expression and its assigned cell type. The
    p-value is the fraction of permutations whose score matched or exceeded
    the observed score, with add-one smoothing so no p-value is ever exactly 0.
    """
    counts = np.zeros_like(observed_scores)
    for _ in range(n_permutations):
        perm_labels = rng.permutation(labels)
        perm_means = _cluster_means(X, perm_labels, categories)
        perm_scores = _lr_scores(perm_means, ligand_idx, receptor_idx)
        counts += (perm_scores >= observed_scores)
    return (counts + 1.0) / (n_permutations + 1.0)


def _to_dense(X) -> np.ndarray:
    if hasattr(X, "toarray"):
        return np.asarray(X.toarray(), dtype=np.float64)
    return np.asarray(X, dtype=np.float64)


def _validate_caps(n_cells: int, n_vars: int, n_permutations: int) -> None:
    if n_cells > caps.MAX_CELLS:
        raise ValueError(f"Dataset has {n_cells} cells; max is {caps.MAX_CELLS}")
    if n_vars > caps.MAX_GENES:
        raise ValueError(f"Dataset has {n_vars} genes; max is {caps.MAX_GENES}")
    if not (0 < n_permutations <= caps.MAX_PERMUTATIONS):
        raise ValueError(
            f"n_permutations must be in (0, {caps.MAX_PERMUTATIONS}], got {n_permutations}"
        )


def infer_cell_communication(
    adata: "ad.AnnData",
    cell_type_key: str,
    lr_pairs: Sequence[Tuple[str, str]],
    n_permutations: int = 200,
    pvalue_threshold: float = 0.05,
    min_cells_per_type: int = 10,
    seed: int = 0,
) -> CellCommResult:
    """Infer significant ligand-receptor interactions between cell types.

    Raises:
        ValueError: on any invalid input (missing column, too-small cell
            types, size caps exceeded, no usable LR genes).
    """
    _validate_caps(adata.n_obs, adata.n_vars, n_permutations)

    if cell_type_key not in adata.obs.columns:
        raise ValueError(f"cell_type_key '{cell_type_key}' not found in adata.obs")

    labels_all = adata.obs[cell_type_key].astype(str).to_numpy()
    categories = sorted(set(labels_all))
    if len(categories) < 2:
        raise ValueError("At least 2 distinct cell types are required")
    if len(categories) > caps.MAX_CLUSTERS:
        raise ValueError(f"Too many cell types ({len(categories)}); max is {caps.MAX_CLUSTERS}")

    counts_per_type: Dict[str, int] = {c: int((labels_all == c).sum()) for c in categories}
    too_small = [c for c, n in counts_per_type.items() if n < min_cells_per_type]
    if too_small:
        raise ValueError(
            f"Cell type(s) {too_small} have fewer than min_cells_per_type={min_cells_per_type} cells"
        )

    var_names = set(adata.var_names)
    usable_pairs = [(l, r) for l, r in lr_pairs if l in var_names and r in var_names]
    if not usable_pairs:
        raise ValueError(
            "None of the supplied ligand-receptor pairs' genes are present in the dataset"
        )

    lr_genes = sorted({g for pair in usable_pairs for g in pair})
    gene_index = {g: i for i, g in enumerate(lr_genes)}
    gene_positions = adata.var_names.get_indexer(lr_genes)
    x_lr = _to_dense(adata.X[:, gene_positions])   # (n_cells, n_lr_genes)

    ligand_idx = np.array([gene_index[l] for l, _ in usable_pairs])
    receptor_idx = np.array([gene_index[r] for _, r in usable_pairs])

    observed_means = _cluster_means(x_lr, labels_all, categories)
    observed_scores = _lr_scores(observed_means, ligand_idx, receptor_idx)

    rng = np.random.default_rng(seed)
    pvalues = _permutation_pvalues(
        x_lr, labels_all, categories, ligand_idx, receptor_idx,
        observed_scores, n_permutations, rng,
    )

    interactions: List[InteractionResult] = []
    n_clusters = len(categories)
    for p_i, (ligand, receptor) in enumerate(usable_pairs):
        for a in range(n_clusters):
            for b in range(n_clusters):
                score = float(observed_scores[p_i, a, b])
                pval = float(pvalues[p_i, a, b])
                if score > 0 and pval <= pvalue_threshold:
                    interactions.append(InteractionResult(
                        ligand=ligand, receptor=receptor,
                        source_cluster=categories[a], target_cluster=categories[b],
                        ligand_mean_expression=float(observed_means[a, gene_index[ligand]]),
                        receptor_mean_expression=float(observed_means[b, gene_index[receptor]]),
                        interaction_score=score, p_value=pval,
                    ))

    interactions.sort(key=lambda i: i.p_value)

    return CellCommResult(
        cell_types=categories,
        cell_counts=[counts_per_type[c] for c in categories],
        n_cells=int(len(labels_all)),
        n_genes_tested=len(lr_genes),
        n_permutations=n_permutations,
        pvalue_threshold=pvalue_threshold,
        n_pairs_tested=len(usable_pairs),
        interactions=interactions,
    )
