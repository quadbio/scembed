"""Tests for integration evaluation functionality."""

import numpy as np
import pytest

from scembed.evaluation import IntegrationEvaluator
from scembed.methods.cpu_methods import HVGMethod, PrecomputedEmbeddingMethod
from scembed.methods.gpu_methods import HarmonyMethod


@pytest.fixture(scope="module")
def adata(pbmc_data_session):
    """PBMC3k with embeddings from different integration methods, computed once per module."""
    adata = pbmc_data_session.copy()

    # Method 1: Precomputed (uses existing PCA)
    pca_method = PrecomputedEmbeddingMethod(adata, embedding_key="X_pca")
    pca_method.fit_transform()
    adata.obsm["X_pca"] = pca_method.adata.obsm["X_pca"]

    # Method 2: HVG-based embedding
    hvg_method = HVGMethod(adata, n_top_genes=500, embedding_key="X_hvg")
    hvg_method.fit_transform()
    adata.obsm["X_hvg"] = hvg_method.adata.obsm["X_hvg"]

    # Method 3: Harmony (GPU method)
    harmony_method = HarmonyMethod(adata, embedding_key="X_harmony")
    harmony_method.fit_transform()
    adata.obsm["X_harmony"] = harmony_method.adata.obsm["X_harmony"]

    return adata


@pytest.fixture(scope="module")
def evaluated(adata):
    """Evaluator with a full scIB benchmark at default settings, shared by the tests that read its results."""
    evaluator = IntegrationEvaluator(
        adata=adata,
        embedding_key="X_harmony",
        batch_key="batch",
        cell_type_key="cell_type",
        baseline_embedding_key="X_pca",
    )
    evaluator.evaluate_scib()
    return evaluator


def _check_scib_metrics(evaluator):
    """Check that the evaluator holds numeric scIB metrics for its embedding."""
    assert evaluator.scib_metrics is not None
    assert isinstance(evaluator.scib_metrics, type(evaluator.scib_metrics).__bases__[0])  # DataFrame-like
    assert evaluator.embedding_key in evaluator.scib_metrics.index

    metrics = evaluator.scib_metrics.loc[evaluator.embedding_key]
    assert len(metrics) > 0
    assert all(isinstance(val, int | float | np.number) for val in metrics.values)


class TestIntegrationEvaluator:
    """Test suite for IntegrationEvaluator following typical evaluation pipeline."""

    @pytest.mark.parametrize(
        "embedding_key,baseline_key",
        [
            ("X_pca", "X_pca_unintegrated"),
            ("X_hvg", "X_pca_unintegrated"),
            ("X_harmony", "X_pca_unintegrated"),
        ],
    )
    def test_evaluator_initialization(self, adata, embedding_key, baseline_key):
        """Test IntegrationEvaluator initialization with different embedding keys."""
        evaluator = IntegrationEvaluator(
            adata=adata,
            embedding_key=embedding_key,
            batch_key="batch",
            cell_type_key="cell_type",
            baseline_embedding_key=baseline_key,
        )

        # Check basic attributes
        assert evaluator.embedding_key == embedding_key
        assert evaluator.batch_key == "batch"
        assert evaluator.cell_type_key == "cell_type"
        assert evaluator.baseline_embedding_key == baseline_key

        # Check that adata is copied and has the required embedding
        assert evaluator.adata is not adata  # Should be a copy
        assert embedding_key in evaluator.adata.obsm
        assert evaluator.adata.n_obs > 0
        assert evaluator.adata.n_vars > 0

        # Check that baseline embedding exists (created automatically if missing)
        assert baseline_key in evaluator.adata.obsm

        # Check output directory setup
        assert evaluator.output_dir.exists()
        assert evaluator.figures_dir.exists()
        assert evaluator.figures_dir.parent == evaluator.output_dir

        # Check initial state
        assert evaluator.scib_metrics is None

    def test_evaluate_scib(self, evaluated):
        """Test scIB evaluation with default settings."""
        _check_scib_metrics(evaluated)

    def test_evaluate_scib_subsampled(self, adata):
        """Test scIB evaluation with subsampling and min-max scaling."""
        evaluator = IntegrationEvaluator(
            adata=adata,
            embedding_key="X_pca",
            batch_key="batch",
            cell_type_key="cell_type",
        )
        evaluator.evaluate_scib(subsample_to=500, min_max_scale=True)

        _check_scib_metrics(evaluator)

    def test_compute_and_show_embeddings(self, adata):
        """Test UMAP computation and visualization."""
        key_added = "X_umap_test"
        evaluator = IntegrationEvaluator(
            adata=adata,
            embedding_key="X_pca",
            batch_key="batch",
            cell_type_key="cell_type",
        )

        # Compute and visualize embeddings
        evaluator.compute_and_show_embeddings(
            key_added=key_added,
            use_rapids=False,  # rapids would require a GPU
            additional_colors=["batch"],  # Test additional coloring
        )

        # Check that UMAP embedding was computed
        assert key_added in evaluator.adata.obsm
        assert evaluator.adata.obsm[key_added].shape[0] == evaluator.adata.n_obs
        assert evaluator.adata.obsm[key_added].shape[1] == 2  # UMAP is 2D

        # Check that the figure was saved
        expected_figure = evaluator.figures_dir / "umap_evaluation.png"
        assert expected_figure.exists()
        assert expected_figure.stat().st_size > 0  # File should not be empty

    def test_get_summary_metrics(self, adata, evaluated):
        """Test extraction of summary metrics after evaluation."""
        evaluator = IntegrationEvaluator(
            adata=adata,
            embedding_key="X_hvg",
            batch_key="batch",
            cell_type_key="cell_type",
        )

        # Should raise error before evaluation
        with pytest.raises(ValueError, match="Run evaluate_scib"):
            evaluator.get_summary_metrics()

        # After evaluation, should return metrics
        summary = evaluated.get_summary_metrics()
        assert isinstance(summary, dict)
        assert len(summary) > 0

        # All values should be numeric
        assert all(isinstance(val, int | float | np.number) for val in summary.values())

    def test_typical_evaluation_pipeline(self, evaluated):
        """Test the complete evaluation pipeline as used in practice, after the scIB step in `evaluated`."""
        # Get summary metrics (as would be logged to wandb)
        summary = evaluated.get_summary_metrics()
        assert isinstance(summary, dict)
        assert len(summary) > 0

        # Generate UMAP plots
        evaluated.compute_and_show_embeddings(wspace=0.7)

        # Check that all expected outputs are present
        assert "X_umap" in evaluated.adata.obsm
        assert (evaluated.figures_dir / "umap_evaluation.png").exists()

        # Check that we can access the embedding key in results
        assert evaluated.embedding_key in evaluated.scib_metrics.index

    def test_ignore_cell_types_functionality(self, adata):
        """Test filtering of specific cell types during evaluation."""
        # Add some cells with unknown annotation
        adata_with_unknown = adata.copy()

        # Add "unknown" category to the categorical cell_type column
        adata_with_unknown.obs["cell_type"] = adata_with_unknown.obs["cell_type"].cat.add_categories(["unknown"])

        unknown_mask = np.random.choice([True, False], size=adata_with_unknown.n_obs, p=[0.1, 0.9])
        adata_with_unknown.obs.loc[unknown_mask, "cell_type"] = "unknown"

        # Test ignoring unknown cell types
        evaluator = IntegrationEvaluator(
            adata=adata_with_unknown,
            embedding_key="X_pca",
            batch_key="batch",
            cell_type_key="cell_type",
            ignore_cell_types=["unknown"],
        )

        # Should have fewer cells after filtering
        expected_cells = (~unknown_mask).sum()
        assert evaluator.adata.n_obs == expected_cells

        # Should not contain ignored cell types
        assert "unknown" not in evaluator.adata.obs["cell_type"].values

    def test_evaluator_repr(self, adata, evaluated):
        """Test string representation of the evaluator."""
        evaluator = IntegrationEvaluator(
            adata=adata,
            embedding_key="X_pca",
            batch_key="batch",
            cell_type_key="cell_type",
        )

        repr_str = repr(evaluator)

        # Should contain key information
        assert "IntegrationEvaluator" in repr_str
        assert "X_pca" in repr_str
        assert "batch" in repr_str
        assert "cell_type" in repr_str
        assert "not evaluated" in repr_str

        # After evaluation, should show evaluated status
        repr_str_after = repr(evaluated)
        assert "evaluated" in repr_str_after
        assert "not evaluated" not in repr_str_after
