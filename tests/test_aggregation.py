"""Tests for scIB aggregation functionality."""

import pandas as pd
import pytest

from scembed.aggregation import scIBAggregator


class TestscIBAggregator:
    """Test suite for scIBAggregator following typical usage pipeline."""

    @pytest.fixture(autouse=True)
    def setup_aggregator(self):
        """Set up aggregator instance for testing."""
        # Using the same entity/project from the example notebook
        self.agg = scIBAggregator(entity="spatial_vi", project="scembed_test_3")

    def test_initialization(self):
        """Test scIBAggregator initialization."""
        assert self.agg.entity == "spatial_vi"
        assert self.agg.project == "scembed_test_3"
        assert self.agg.output_dir.exists()
        assert self.agg.raw_df is None
        assert self.agg.method_data == {}
        assert self.agg.results is None
        assert self.agg.n_runs_fetched == 0

    def test_fetch_runs(self):
        """Test fetching runs from wandb."""
        self.agg.fetch_runs()

        # Should have fetched some runs
        assert self.agg.raw_df is not None
        assert self.agg.n_runs_fetched > 0
        assert len(self.agg.method_data) > 0
        assert len(self.agg.available_scib_metrics) > 0

        # Check that method data structure is correct
        for _method, data in self.agg.method_data.items():
            assert "configs" in data
            assert "scib_benchmarker" in data
            assert "other_logs" in data

    @pytest.mark.parametrize("method", ["harmony", "scvi"])
    def test_get_method_runs(self, method):
        """Test retrieving runs for specific methods."""
        self.agg.fetch_runs()

        if method in self.agg.available_methods:
            method_data = self.agg.get_method_runs(method)

            assert "configs" in method_data
            assert "scib_benchmarker" in method_data
            assert "other_logs" in method_data

            # Check that data is sorted (best runs first)
            configs_df = method_data["configs"]
            assert isinstance(configs_df, pd.DataFrame)
            assert len(configs_df) > 0

    @pytest.mark.parametrize("sort_by", ["Total", "ARI"])
    def test_aggregate(self, sort_by):
        """Test aggregating best runs per method."""
        self.agg.fetch_runs()

        # Check if the sort_by metric is available
        if sort_by in self.agg.available_scib_metrics:
            self.agg.aggregate(sort_by=sort_by)

            assert self.agg.results is not None
            assert "configs" in self.agg.results
            assert "scib_benchmarker" in self.agg.results
            assert "other_logs" in self.agg.results

            # Should have one entry per method
            configs_df = self.agg.results["configs"]
            assert isinstance(configs_df, pd.DataFrame)
            assert len(configs_df) > 0
            assert len(configs_df) == len(self.agg.available_methods)

    def test_get_models_and_embeddings(self):
        """Test downloading models and embeddings for best runs."""
        self.agg.fetch_runs()
        self.agg.aggregate()

        # This should run without errors
        self.agg.get_models_and_embeddings()

        # Check that method directories were created
        assert self.agg.results is not None
        configs_df = self.agg.results["configs"]
        assert isinstance(configs_df, pd.DataFrame)

        for method in configs_df.index:
            row = configs_df.loc[method]
            run_id = row["run_id"]
            # Method directories are created with format {method}_{run_id}
            method_dir = self.agg.output_dir / f"{method}_{run_id}"
            assert method_dir.exists()

            # Check subdirectories exist
            assert (method_dir / "models").exists()
            assert (method_dir / "embeddings").exists()

    def test_available_methods_property(self):
        """Test available_methods property."""
        # Before fetching - should be empty
        assert self.agg.available_methods == []

        # After fetching - should have methods
        self.agg.fetch_runs()
        assert len(self.agg.available_methods) > 0
        assert isinstance(self.agg.available_methods, list)

    def test_repr(self):
        """Test string representation."""
        # Before fetching
        repr_str = repr(self.agg)
        assert "spatial_vi/scembed_test_3" in repr_str
        assert "no data fetched" in repr_str

        # After fetching
        self.agg.fetch_runs()
        repr_str_after = repr(self.agg)
        assert "spatial_vi/scembed_test_3" in repr_str_after
        assert "runs" in repr_str_after
        assert "methods" in repr_str_after
