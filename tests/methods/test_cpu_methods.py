"""Tests for CPU-based integration methods."""

import numpy as np
import pytest

from scembed.methods.cpu_methods import HVGMethod, LIGERMethod, PrecomputedEmbeddingMethod, ScanoramaMethod


class TestPrecomputedEmbeddingMethod:
    """Test suite for PrecomputedEmbeddingMethod."""

    @pytest.mark.parametrize("embedding_key", ["X_pca"])
    def test_precomputed_embedding_method(self, pbmc_data, embedding_key):
        """Test precomputed embedding method with different embedding keys."""
        method = PrecomputedEmbeddingMethod(pbmc_data, embedding_key=embedding_key)

        assert method.embedding_key == embedding_key
        assert method.source_embedding_key == embedding_key
        assert not method.is_fitted

        method.fit_transform()

        assert method.is_fitted
        assert method.embedding_key in method.adata.obsm
        assert method.adata.obsm[method.embedding_key].shape[0] == method.adata.n_obs
        # Should be the same as the original embedding
        np.testing.assert_array_equal(
            method.adata.obsm[method.embedding_key], method.adata.obsm[method.source_embedding_key]
        )


class TestLIGERMethod:
    """Test suite for LIGERMethod."""

    @pytest.mark.parametrize(
        "k,value_lambda,max_iters",
        [
            (5, None, 5),  # Reduced k and max_iters for speed
            (10, 5.0, 10),  # Reduced iterations
            (8, 10.0, 8),  # Reduced iterations
        ],
    )
    def test_liger_method(self, pbmc_data, k, value_lambda, max_iters):
        """Test LIGER method with different parameter combinations."""
        pytest.importorskip("pyliger")

        method = LIGERMethod(pbmc_data, k=k, value_lambda=value_lambda, max_iters=max_iters)

        assert method.k == k
        assert method.value_lambda == value_lambda
        assert method.max_iters == max_iters
        assert not method.is_fitted

        method.fit_transform()

        assert method.is_fitted
        assert method.embedding_key in method.adata.obsm
        assert method.adata.obsm[method.embedding_key].shape[0] == method.adata.n_obs
        # LIGER should produce k-dimensional embedding
        assert method.adata.obsm[method.embedding_key].shape[1] == k


class TestHVGMethod:
    """Test suite for HVGMethod."""

    @pytest.mark.parametrize(
        "n_top_genes,flavor,scale",
        [
            (500, "cell_ranger", False),  # Reduced from 1000
            (1000, "seurat", True),  # Reduced from 2000
            (300, "cell_ranger", False),  # Reduced from 500
        ],
    )
    def test_hvg_method(self, pbmc_data, n_top_genes, flavor, scale):
        """Test HVG method with different parameter combinations."""
        method = HVGMethod(pbmc_data, n_top_genes=n_top_genes, flavor=flavor, scale=scale)

        assert method.n_top_genes == n_top_genes
        assert method.flavor == flavor
        assert method.scale == scale
        assert not method.is_fitted

        method.fit_transform()

        assert method.is_fitted
        assert method.embedding_key in method.adata.obsm
        assert method.adata.obsm[method.embedding_key].shape[0] == method.adata.n_obs
        # After HVG selection, should have at most n_top_genes features
        assert method.adata.obsm[method.embedding_key].shape[1] <= n_top_genes
        # Should have exactly n_top_genes after selection (unless fewer genes available)
        expected_genes = min(n_top_genes, pbmc_data.n_vars)
        assert method.adata.obsm[method.embedding_key].shape[1] == expected_genes


class TestScanoramaMethod:
    """Test suite for ScanoramaMethod."""

    @pytest.mark.parametrize(
        "knn,alpha,sigma",
        [
            (10, None, None),  # Reduced from 20
            (5, 0.10, 15.0),  # Reduced from 10
            (15, 0.05, 20.0),  # Reduced from 30
        ],
    )
    def test_scanorama_method(self, pbmc_data, knn, alpha, sigma):
        """Test Scanorama method with different parameter combinations."""
        pytest.importorskip("scanorama")

        method = ScanoramaMethod(pbmc_data, knn=knn, alpha=alpha, sigma=sigma)

        assert method.knn == knn
        assert method.alpha == alpha
        assert method.sigma == sigma
        assert not method.is_fitted

        method.fit_transform()

        assert method.is_fitted
        assert method.embedding_key in method.adata.obsm
        assert method.adata.obsm[method.embedding_key].shape[0] == method.adata.n_obs
        # Scanorama typically produces embeddings with same dimensionality as input or reduced
        assert method.adata.obsm[method.embedding_key].shape[1] > 0
        # Scanorama typically produces embeddings with same dimensionality as input or reduced
        assert method.adata.obsm[method.embedding_key].shape[1] > 0
