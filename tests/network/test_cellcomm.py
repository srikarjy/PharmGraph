"""Tests for the cell-cell communication permutation-test algorithm."""

import numpy as np
import pandas as pd
import pytest

import anndata as ad

from src.network import LIGAND_RECEPTOR_PAIRS, LR_PAIRS, generate_demo_dataset
from src.network.cellcomm import (
    CellCommResult,
    _cluster_means,
    _lr_scores,
    infer_cell_communication,
)
from src.network import caps as network_caps


def make_adata(X: np.ndarray, labels, genes) -> ad.AnnData:
    obs = pd.DataFrame({"cell_type": pd.Categorical(labels)})
    obs.index = [f"cell_{i}" for i in range(len(labels))]
    var = pd.DataFrame(index=list(genes))
    return ad.AnnData(X=X.astype(np.float32), obs=obs, var=var)


class TestClusterMeans:
    """Verify the vectorized indicator-matmul groupby against a naive loop."""

    def test_matches_naive_groupby(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(60, 5))
        labels = rng.choice(["a", "b", "c"], size=60)
        categories = sorted(set(labels))

        result = _cluster_means(X, labels, categories)
        naive = np.array([X[labels == c].mean(axis=0) for c in categories])

        assert np.allclose(result, naive)


class TestLrScoreFormula:
    """Verify score[p, A, B] = (mean_L[A] + mean_R[B]) / 2 exactly."""

    def test_hand_computed_example(self):
        # 2 clusters, 2 genes: cluster 0 = [10, 2], cluster 1 = [4, 8]
        cluster_means = np.array([[10.0, 2.0], [4.0, 8.0]])
        ligand_idx = np.array([0])   # gene 0 is the ligand
        receptor_idx = np.array([1])  # gene 1 is the receptor

        scores = _lr_scores(cluster_means, ligand_idx, receptor_idx)

        assert scores.shape == (1, 2, 2)
        assert scores[0, 0, 0] == pytest.approx((10.0 + 2.0) / 2)   # A=0 -> B=0
        assert scores[0, 0, 1] == pytest.approx((10.0 + 8.0) / 2)   # A=0 -> B=1
        assert scores[0, 1, 0] == pytest.approx((4.0 + 2.0) / 2)    # A=1 -> B=0
        assert scores[0, 1, 1] == pytest.approx((4.0 + 8.0) / 2)    # A=1 -> B=1


class TestPlantedSignalRecovery:
    """A synthetic case with a known signal should be statistically recoverable."""

    def _build_planted_dataset(self):
        rng = np.random.default_rng(1)
        genes = ["LIG_TRUE", "REC_TRUE", "LIG_DECOY", "REC_DECOY"]
        n_per_type = 80
        labels = np.repeat(["Sender", "Receiver"], n_per_type)
        n_cells = len(labels)

        X = rng.gamma(shape=2.0, scale=0.1, size=(n_cells, len(genes))).astype(np.float32)
        # Plant LIG_TRUE high in Sender, REC_TRUE high in Receiver.
        X[labels == "Sender", 0] += rng.gamma(2.0, 4.0, size=n_per_type)
        X[labels == "Receiver", 1] += rng.gamma(2.0, 4.0, size=n_per_type)
        # LIG_DECOY / REC_DECOY are left as pure background noise -- no signal.

        return make_adata(X, labels, genes)

    def test_true_pair_significant_decoy_not(self):
        adata = self._build_planted_dataset()
        pairs = [("LIG_TRUE", "REC_TRUE"), ("LIG_DECOY", "REC_DECOY")]

        result = infer_cell_communication(
            adata, "cell_type", pairs,
            n_permutations=300, pvalue_threshold=0.01, min_cells_per_type=10, seed=2,
        )

        true_hit = [
            i for i in result.interactions
            if i.ligand == "LIG_TRUE" and i.receptor == "REC_TRUE"
            and i.source_cluster == "Sender" and i.target_cluster == "Receiver"
        ]
        decoy_hit = [
            i for i in result.interactions
            if i.ligand == "LIG_DECOY" and i.receptor == "REC_DECOY"
        ]

        assert len(true_hit) == 1
        assert true_hit[0].p_value <= 0.01
        assert decoy_hit == []


class TestNullCase:
    """Fully random data with no cluster structure should yield no significant pairs."""

    def test_no_significant_pairs(self):
        rng = np.random.default_rng(42)
        n_cells, n_genes = 200, 20
        X = rng.gamma(2.0, 0.2, size=(n_cells, n_genes)).astype(np.float32)
        labels = np.tile(["A", "B"], n_cells // 2)
        genes = [f"G{i}" for i in range(n_genes)]
        adata = make_adata(X, labels, genes)
        pairs = [("G0", "G1"), ("G2", "G3"), ("G4", "G5")]

        result = infer_cell_communication(
            adata, "cell_type", pairs,
            n_permutations=500, pvalue_threshold=0.01, min_cells_per_type=10, seed=7,
        )

        assert result.n_significant == 0


class TestPValueBounds:
    """Add-one smoothing must keep every p-value in (0, 1]."""

    def test_pvalues_bounded_and_never_zero(self):
        adata = generate_demo_dataset(seed=0)
        result = infer_cell_communication(
            adata, "cell_type", LIGAND_RECEPTOR_PAIRS,
            n_permutations=50, pvalue_threshold=1.0, min_cells_per_type=10, seed=3,
        )
        assert result.interactions  # threshold=1.0 keeps everything with score > 0
        for interaction in result.interactions:
            assert 0.0 < interaction.p_value <= 1.0


class TestInputValidation:
    """Bad input and cap violations should raise ValueError, not silently proceed."""

    def test_min_cells_per_type_rejects_small_clusters(self):
        rng = np.random.default_rng(0)
        n_a, n_b = 20, 3
        X = rng.normal(size=(n_a + n_b, 4))
        labels = np.array(["A"] * n_a + ["B"] * n_b)
        genes = ["G0", "G1", "G2", "G3"]
        adata = make_adata(X, labels, genes)

        with pytest.raises(ValueError, match="min_cells_per_type"):
            infer_cell_communication(
                adata, "cell_type", [("G0", "G1")],
                n_permutations=10, pvalue_threshold=0.05, min_cells_per_type=10,
            )

    def test_missing_cell_type_key_raises(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(20, 4))
        genes = ["G0", "G1", "G2", "G3"]
        adata = make_adata(X, ["A"] * 10 + ["B"] * 10, genes)

        with pytest.raises(ValueError, match="cell_type_key"):
            infer_cell_communication(
                adata, "does_not_exist", [("G0", "G1")],
                n_permutations=10, pvalue_threshold=0.05, min_cells_per_type=5,
            )

    def test_too_many_cells_raises(self, monkeypatch):
        monkeypatch.setattr(network_caps, "MAX_CELLS", 10)
        rng = np.random.default_rng(0)
        X = rng.normal(size=(20, 4))
        genes = ["G0", "G1", "G2", "G3"]
        adata = make_adata(X, ["A"] * 10 + ["B"] * 10, genes)

        with pytest.raises(ValueError, match="cells"):
            infer_cell_communication(
                adata, "cell_type", [("G0", "G1")],
                n_permutations=10, pvalue_threshold=0.05, min_cells_per_type=5,
            )

    def test_n_permutations_exceeding_max_raises(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(20, 4))
        genes = ["G0", "G1", "G2", "G3"]
        adata = make_adata(X, ["A"] * 10 + ["B"] * 10, genes)

        with pytest.raises(ValueError, match="n_permutations"):
            infer_cell_communication(
                adata, "cell_type", [("G0", "G1")],
                n_permutations=network_caps.MAX_PERMUTATIONS + 1,
                pvalue_threshold=0.05, min_cells_per_type=5,
            )


class TestSyntheticDemoDataset:
    """The bundled demo dataset should be deterministic and well-formed."""

    def test_shape_and_balance(self):
        adata = generate_demo_dataset(seed=0)
        assert adata.n_obs == 400
        counts = adata.obs["cell_type"].value_counts()
        assert set(counts.index) == {"T_cell", "B_cell", "NK_cell", "Monocyte"}
        assert set(counts.values.tolist()) == {100}

    def test_deterministic_given_seed(self):
        a = generate_demo_dataset(seed=0)
        b = generate_demo_dataset(seed=0)
        assert np.array_equal(a.X, b.X)

    def test_different_seed_differs(self):
        a = generate_demo_dataset(seed=0)
        b = generate_demo_dataset(seed=1)
        assert not np.array_equal(a.X, b.X)


class TestBundledLrPairs:
    """The curated ligand-receptor pair list should be well-formed."""

    def test_no_duplicates(self):
        assert len(LIGAND_RECEPTOR_PAIRS) == len(set(LIGAND_RECEPTOR_PAIRS))

    def test_gene_symbols_uppercase(self):
        for ligand, receptor in LIGAND_RECEPTOR_PAIRS:
            assert ligand == ligand.upper()
            assert receptor == receptor.upper()

    def test_sane_count(self):
        assert 40 <= len(LR_PAIRS) <= 100
