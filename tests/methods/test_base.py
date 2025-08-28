"""Tests for BaseIntegrationMethod."""

import h5py
import numpy as np
import pandas as pd
import pytest

from scembed.methods.base import BaseIntegrationMethod


class ConcreteIntegrationMethod(BaseIntegrationMethod):
    """Concrete implementation for testing BaseIntegrationMethod."""

    def fit(self):
        """Simple fit implementation for testing."""
        self.is_fitted = True

    def transform(self):
        """Simple transform implementation for testing."""
        if not self.is_fitted:
            raise ValueError("Method must be fitted before transform")

        # Create a simple 2D embedding
        n_cells = self.adata.n_obs
        embedding = np.random.RandomState(42).randn(n_cells, 2)
        self.adata.obsm[self.embedding_key] = embedding


class TestBaseIntegrationMethod:
    """Test suite for BaseIntegrationMethod."""

    def test_initialization_default(self, pbmc_data):
        """Test default initialization."""
        method = ConcreteIntegrationMethod(pbmc_data)

        assert method.name == "ConcreteIntegration"
        assert method.embedding_key == "X_concreteintegration"
        assert not method.is_fitted
        assert method.batch_key == "batch"
        assert method.cell_type_key == "cell_type"
        assert method.output_dir.exists()
        assert method.models_dir.exists()
        assert method.embedding_dir.exists()

    @pytest.mark.parametrize(
        "batch_key,cell_type_key,expected_name",
        [
            ("batch", "cell_type", "ConcreteIntegration"),
            ("custom_batch", "custom_celltype", "ConcreteIntegration"),
        ],
    )
    def test_initialization_custom_keys(self, pbmc_data, batch_key, cell_type_key, expected_name):
        """Test initialization with custom keys."""
        # Add custom keys to test data
        test_data = pbmc_data.copy()
        if batch_key != "batch":
            test_data.obs[batch_key] = test_data.obs["batch"]
        if cell_type_key != "cell_type":
            test_data.obs[cell_type_key] = test_data.obs["cell_type"]

        method = ConcreteIntegrationMethod(test_data, batch_key=batch_key, cell_type_key=cell_type_key)

        assert method.name == expected_name
        assert method.batch_key == batch_key
        assert method.cell_type_key == cell_type_key

    def test_validation_missing_batch_key(self, pbmc_data):
        """Test validation fails when batch key is missing."""
        test_data = pbmc_data.copy()
        del test_data.obs["batch"]

        with pytest.raises(ValueError, match="Batch key 'batch' not found"):
            ConcreteIntegrationMethod(test_data)

    def test_validation_spatial_data(self, spatial_data):
        """Test spatial data validation."""
        method = ConcreteIntegrationMethod(
            spatial_data, validate_spatial=True, batch_key="batch", cell_type_key="cell_type"
        )
        assert method.spatial_key in method.adata.obsm

    def test_validation_spatial_data_missing_coords(self, pbmc_data):
        """Test spatial validation fails when coordinates are missing."""
        with pytest.raises(ValueError, match="Spatial coordinates not found"):
            ConcreteIntegrationMethod(pbmc_data, validate_spatial=True)

    def test_fit_transform(self, pbmc_data):
        """Test fit_transform workflow."""
        method = ConcreteIntegrationMethod(pbmc_data)

        assert not method.is_fitted
        assert method.embedding_key not in method.adata.obsm

        method.fit_transform()

        assert method.is_fitted
        assert method.embedding_key in method.adata.obsm
        assert method.adata.obsm[method.embedding_key].shape == (method.adata.n_obs, 2)

    @pytest.mark.parametrize(
        "format_type,expected_suffix",
        [
            ("parquet", ".parquet"),
            ("pickle", ".pkl.gz"),
            ("h5", ".h5"),
        ],
    )
    def test_save_embedding_formats(self, pbmc_data, format_type, expected_suffix):
        """Test saving embeddings in different formats."""
        method = ConcreteIntegrationMethod(pbmc_data)
        method.fit_transform()

        saved_path = method.save_embedding(format_type=format_type)

        assert saved_path.exists()
        assert saved_path.name.endswith(expected_suffix)

        # Verify file contains expected data
        if format_type == "parquet":
            df = pd.read_parquet(saved_path)
            assert df.shape == (method.adata.n_obs, 2)
            assert list(df.index) == list(method.adata.obs_names)
        elif format_type == "h5":
            with h5py.File(saved_path, "r") as f:
                assert "embedding" in f
                assert "cell_names" in f
                assert "dim_names" in f

    def test_save_embedding_before_fit(self, pbmc_data):
        """Test saving embedding fails before fitting."""
        method = ConcreteIntegrationMethod(pbmc_data)

        with pytest.raises(ValueError, match="Method must be fitted"):
            method.save_embedding()

    def test_get_model_info(self, pbmc_data):
        """Test model info retrieval."""
        method = ConcreteIntegrationMethod(pbmc_data, custom_param=123)
        info = method.get_model_info()

        assert info["method"] == "ConcreteIntegration"
        assert info["params"] == {"custom_param": 123}
        assert not info["is_fitted"]
        assert info["embedding_key"] == "X_concreteintegration"

        method.fit()
        info_fitted = method.get_model_info()
        assert info_fitted["is_fitted"]

    def test_repr(self, pbmc_data):
        """Test string representation."""
        method = ConcreteIntegrationMethod(pbmc_data, test_param="value")
        repr_str = repr(method)

        assert "ConcreteIntegrationMethod" in repr_str
        assert "test_param=value" in repr_str
        assert "not fitted" in repr_str
        assert f"{pbmc_data.n_obs:,} cells" in repr_str

        method.fit()
        fitted_repr = repr(method)
        assert "fitted" in fitted_repr

    def test_cell_type_handling_with_missing_values(self, pbmc_data):
        """Test handling of missing cell type values."""
        test_data = pbmc_data.copy()
        # Introduce some missing values
        test_data.obs.loc[test_data.obs.index[:10], "cell_type"] = pd.NA

        method = ConcreteIntegrationMethod(test_data)

        # Check that missing values were converted to unlabeled category
        assert method.unlabeled_category in method.adata.obs["cell_type"].cat.categories
        unlabeled_count = (method.adata.obs["cell_type"] == method.unlabeled_category).sum()
        assert unlabeled_count == 10

    @pytest.mark.parametrize("use_hvg", [True, False])
    def test_hvg_handling(self, pbmc_data, use_hvg):
        """Test highly variable genes handling."""
        method = ConcreteIntegrationMethod(pbmc_data, use_hvg=use_hvg)
        assert method.use_hvg == use_hvg

        if use_hvg:
            assert method.hvg_key in method.adata.var.columns
