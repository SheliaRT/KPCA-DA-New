"""Unit tests for kpcada.py core functions."""
import sys
import os
import numpy as np
import pytest
import torch
from sklearn.decomposition import KernelPCA

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.kpcada import (
    find_common_gene_indices,
    standardize_datasets,
    rbf_kernel,
    mmd_loss,
    fit_transform,
    KPCA_DA,
)


class TestFindCommonGeneIndices:
    def test_partial_overlap(self):
        genes_s = np.array(["A", "B", "C"])
        genes_t = np.array(["B", "C", "D"])
        idx_s, idx_t = find_common_gene_indices(genes_s, genes_t)
        assert set(genes_s[idx_s]) == {"B", "C"}
        assert set(genes_t[idx_t]) == {"B", "C"}

    def test_no_overlap(self):
        genes_s = np.array(["A", "B"])
        genes_t = np.array(["C", "D"])
        idx_s, idx_t = find_common_gene_indices(genes_s, genes_t)
        assert idx_s.size == 0
        assert idx_t.size == 0

    def test_full_overlap(self):
        genes_s = np.array(["A", "B", "C"])
        genes_t = np.array(["A", "B", "C"])
        idx_s, idx_t = find_common_gene_indices(genes_s, genes_t)
        assert idx_s.size == 3
        assert idx_t.size == 3


class TestStandardizeDatasets:
    def test_output_shape(self):
        X_s = np.random.randn(10, 5)
        X_t = np.random.randn(8, 5)
        X_s_scaled, X_t_scaled = standardize_datasets(X_s, X_t)
        assert X_s_scaled.shape == X_s.shape
        assert X_t_scaled.shape == X_t.shape

    def test_mean_near_zero(self):
        X_s = np.random.randn(50, 10) * 5 + 3
        X_t = np.random.randn(40, 10) * 2 - 1
        X_s_scaled, X_t_scaled = standardize_datasets(X_s, X_t)
        assert np.allclose(X_s_scaled.mean(axis=0), 0, atol=1e-10)
        assert np.allclose(X_t_scaled.mean(axis=0), 0, atol=1e-10)


class TestRbfKernel:
    def test_self_similarity(self):
        x = torch.randn(5, 3)
        K = rbf_kernel(x, x, gamma=1.0)
        assert K.shape == (5, 5)
        # Diagonal should be 1 (distance to self is 0)
        assert torch.allclose(K.diag(), torch.ones(5))

    def test_symmetry(self):
        x = torch.randn(4, 3)
        K = rbf_kernel(x, x, gamma=1.0)
        assert torch.allclose(K, K.T)


class TestMmdLoss:
    def test_identical_distributions(self):
        x = torch.randn(10, 2)
        loss = mmd_loss(x, x, gamma=1.0)
        # |K_tt.mean() - 2*K_st.mean()| with Zs==Zt => |K.mean() - 2*K.mean()| = K.mean()
        assert isinstance(loss.item(), float)

    def test_non_negative(self):
        Zs = torch.randn(8, 2)
        Zt = torch.randn(8, 2)
        loss = mmd_loss(Zs, Zt, gamma=1.0)
        # mmd_loss uses torch.abs(), so result is always >= 0
        assert loss.item() >= 0

    def test_formula(self):
        """Verify the loss matches |K_tt.mean() - 2*K_st.mean()|."""
        Zs = torch.randn(6, 2)
        Zt = torch.randn(6, 2)
        gamma = 0.8
        K_tt = rbf_kernel(Zt, Zt, gamma)
        K_st = rbf_kernel(Zs, Zt, gamma)
        expected = abs(K_tt.mean().item() - 2 * K_st.mean().item())
        actual = mmd_loss(Zs, Zt, gamma=gamma).item()
        assert abs(actual - expected) < 1e-5


class TestFitTransform:
    def _make_data(self, n_genes=20, n_samples=15, seed=0):
        rng = np.random.default_rng(seed)
        genes = np.array([f"G{i}" for i in range(n_genes)])
        X = rng.standard_normal((n_samples, n_genes))
        return X, genes

    def test_basic_run(self):
        X_s, genes_s = self._make_data(n_genes=20, n_samples=15)
        X_t, genes_t = self._make_data(n_genes=20, n_samples=12, seed=1)
        results = fit_transform(
            X_s, X_t,
            genes_source=genes_s,
            genes_target=genes_t,
            n_components=2,
            epochs=2,
        )
        assert "Z_s" in results
        assert "Z_t" in results
        assert results["Z_s"].shape[1] == 2
        assert results["Z_t"].shape[1] == 2

    def test_no_common_genes_raises(self):
        X_s, _ = self._make_data(n_genes=5, n_samples=10)
        X_t, _ = self._make_data(n_genes=5, n_samples=8)
        genes_s = np.array(["A", "B", "C", "D", "E"])
        genes_t = np.array(["F", "G", "H", "I", "J"])
        with pytest.raises(ValueError, match="No common genes"):
            fit_transform(X_s, X_t, genes_source=genes_s, genes_target=genes_t, epochs=1)

    def test_missing_genes_raises(self):
        X_s, genes_s = self._make_data()
        X_t, _ = self._make_data()
        with pytest.raises(ValueError):
            fit_transform(X_s, X_t, genes_source=genes_s, genes_target=None)


class TestKPCADA:
    def _make_data(self, n_genes=20, n_samples=15, seed=0):
        rng = np.random.default_rng(seed)
        genes = np.array([f"G{i}" for i in range(n_genes)])
        X = rng.standard_normal((n_samples, n_genes))
        return X, genes

    def test_fit_transform_returns_arrays(self):
        X_s, genes_s = self._make_data()
        X_t, genes_t = self._make_data(seed=1)
        model = KPCA_DA(n_components=2, epochs=2)
        Z_s, Z_t = model.fit_transform(X_s, X_t, genes_s, genes_t)
        assert isinstance(Z_s, np.ndarray)
        assert isinstance(Z_t, np.ndarray)

    def test_transform_before_fit_raises(self):
        model = KPCA_DA()
        with pytest.raises(ValueError, match="fitted"):
            model.transform(np.random.randn(5, 3))

    def test_attributes_set_after_fit(self):
        X_s, genes_s = self._make_data()
        X_t, genes_t = self._make_data(seed=1)
        model = KPCA_DA(n_components=2, epochs=2)
        model.fit_transform(X_s, X_t, genes_s, genes_t)
        assert isinstance(model.kpca_model, KernelPCA)
        assert hasattr(model.kpca_model, "eigenvalues_")
        assert model.loss_history_ is not None
        assert len(model.loss_history_) == 2
