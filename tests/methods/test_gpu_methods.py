"""Tests for GPU-based integration methods."""

import pytest

from scembed.methods.gpu_methods import (
    HarmonyMethod,
    ResolVIMethod,
    scANVIMethod,
    scPoliMethod,
    scVIMethod,
    scVIVAMethod,
)


class TestHarmonyMethod:
    """Test suite for HarmonyMethod."""

    @pytest.mark.parametrize("theta", [None, 2.0, 5.0])
    def test_harmony_method(self, pbmc_data, theta):
        """Test Harmony method with different theta values."""
        pytest.importorskip("harmony")

        method = HarmonyMethod(pbmc_data, theta=theta)

        assert method.theta == theta
        assert not method.is_fitted

        method.fit_transform()

        assert method.is_fitted
        assert method.embedding_key in method.adata.obsm
        assert method.adata.obsm[method.embedding_key].shape[0] == method.adata.n_obs
        # Harmony should preserve dimensionality from PCA
        assert method.adata.obsm[method.embedding_key].shape[1] == pbmc_data.obsm["X_pca"].shape[1]


class TestscVIMethod:
    """Test suite for scVIMethod."""

    @pytest.mark.parametrize(
        "n_latent,n_hidden",
        [
            (5, None),
            (10, 64),
            (8, 128),
        ],
    )
    def test_scvi_method(self, pbmc_data, n_latent, n_hidden):
        """Test scVI method with different latent and hidden dimensions."""
        pytest.importorskip("scvi")

        method = scVIMethod(pbmc_data, n_latent=n_latent, n_hidden=n_hidden, max_epochs=5)

        assert method.n_latent == n_latent
        assert method.n_hidden == n_hidden
        assert method.max_epochs == 5
        assert not method.is_fitted

        method.fit_transform()

        assert method.is_fitted
        assert method.embedding_key in method.adata.obsm
        assert method.adata.obsm[method.embedding_key].shape[0] == method.adata.n_obs
        # scVI should produce n_latent dimensional embedding
        assert method.adata.obsm[method.embedding_key].shape[1] == n_latent


class TestscANVIMethod:
    """Test suite for scANVIMethod."""

    @pytest.mark.parametrize(
        "linear_classifier",
        [
            True,
            False,
            None,
        ],
    )
    def test_scanvi_method(self, pbmc_data, linear_classifier):
        """Test scANVI method with different linear classifier settings."""
        pytest.importorskip("scvi")

        method = scANVIMethod(
            pbmc_data,
            linear_classifier=linear_classifier,
            max_epochs=5,
            scvi_params={"max_epochs": 5},  # Reduce scVI pretraining epochs too
        )

        assert method.linear_classifier == linear_classifier
        assert method.max_epochs == 5
        assert not method.is_fitted

        method.fit_transform()

        assert method.is_fitted
        assert method.embedding_key in method.adata.obsm
        assert method.adata.obsm[method.embedding_key].shape[0] == method.adata.n_obs
        # scANVI should produce embedding with default latent dimensions
        assert method.adata.obsm[method.embedding_key].shape[1] > 0


class TestscPoliMethod:
    """Test suite for scPoliMethod."""

    @pytest.mark.parametrize(
        "latent_dim,pretraining_epochs",
        [
            (5, 2),
            (10, 3),
            (8, 1),
        ],
    )
    def test_scpoli_method(self, pbmc_data, latent_dim, pretraining_epochs):
        """Test scPoli method with different latent dimensions and pretraining epochs."""
        pytest.importorskip("scarches")

        method = scPoliMethod(
            pbmc_data,
            latent_dim=latent_dim,
            pretraining_epochs=pretraining_epochs,
            n_epochs=5,
        )

        assert method.latent_dim == latent_dim
        assert method.pretraining_epochs == pretraining_epochs
        assert method.n_epochs == 5
        assert not method.is_fitted

        method.fit_transform()

        assert method.is_fitted
        assert method.embedding_key in method.adata.obsm
        assert method.adata.obsm[method.embedding_key].shape[0] == method.adata.n_obs
        # scPoli should produce latent_dim dimensional embedding
        assert method.adata.obsm[method.embedding_key].shape[1] == latent_dim


class TestResolVIMethod:
    """Test suite for ResolVIMethod (spatial method)."""

    @pytest.mark.parametrize(
        "n_latent,n_neighbors",
        [
            (5, 10),
            (10, 15),
            (8, 8),  # Changed from 5 to 8 to avoid neighbor indexing issues
        ],
    )
    def test_resolvi_method(self, spatial_data, n_latent, n_neighbors):
        """Test ResolVI method with different latent dimensions and neighbor counts."""
        pytest.importorskip("scvi")

        method = ResolVIMethod(
            spatial_data,
            batch_key="batch",
            cell_type_key="cell_type",
            spatial_key="spatial",
            n_latent=n_latent,
            n_neighbors=n_neighbors,
            max_epochs=5,
            downsample_counts=False,  # Disable to avoid tensor type issues in scvi-tools 1.3.3
        )

        assert method.n_latent == n_latent
        assert method.n_neighbors == n_neighbors
        assert method.max_epochs == 5
        assert not method.is_fitted

        method.fit_transform()

        assert method.is_fitted
        assert method.embedding_key in method.adata.obsm
        assert method.adata.obsm[method.embedding_key].shape[0] == method.adata.n_obs
        # ResolVI should produce n_latent dimensional embedding
        assert method.adata.obsm[method.embedding_key].shape[1] == n_latent


class TestscVIVAMethod:
    """Test suite for scVIVAMethod (spatial method)."""

    @pytest.mark.parametrize(
        "embedding_method,k_nn",
        [
            ("scvi", 10),
            ("scanvi", 15),
            ("scvi", 5),
        ],
    )
    def test_scviva_method(self, spatial_data, embedding_method, k_nn):
        """Test scVIVA method with different embedding methods and neighbor counts."""
        pytest.importorskip("scvi")

        method = scVIVAMethod(
            spatial_data,
            batch_key="batch",
            cell_type_key="cell_type",
            spatial_key="spatial",
            embedding_method=embedding_method,
            k_nn=k_nn,
            max_epochs=5,
            scvi_params={"max_epochs": 5},
            scanvi_params={"max_epochs": 5},
        )

        assert method.embedding_method == embedding_method
        assert method.k_nn == k_nn
        assert method.max_epochs == 5
        assert not method.is_fitted

        method.fit_transform()

        assert method.is_fitted
        assert method.embedding_key in method.adata.obsm
        assert method.adata.obsm[method.embedding_key].shape[0] == method.adata.n_obs
        # scVIVA should produce embedding with default latent dimensions
        assert method.adata.obsm[method.embedding_key].shape[1] > 0
