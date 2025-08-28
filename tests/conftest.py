import numpy as np
import pytest
import scanpy as sc
from anndata import AnnData


def _add_categorical_annotation(adata: AnnData, key: str, labels: list[str], seed: int = 42) -> None:
    """Add artificial categorical annotations to AnnData object for testing.

    Parameters
    ----------
    adata
        AnnData object to modify in-place.
    key
        Key name for the annotation.
    labels
        List of category labels to choose from.
    seed
        Random seed for reproducibility.
    """
    np.random.seed(seed)
    annotations = np.random.choice(labels, size=adata.n_obs)
    adata.obs[key] = annotations
    adata.obs[key] = adata.obs[key].astype("category")


@pytest.fixture
def pbmc_data() -> AnnData:
    """PBMC3k dataset for single-cell integration method testing.

    Peripheral blood mononuclear cells from 10X Genomics (2,700 cells × 32,738 genes).
    After filtering: ~2,700 cells × ~13,714 genes. Includes standard preprocessing:
    normalization, log-transformation, HVG selection, and PCA.

    Returns
    -------
    AnnData
        Preprocessed PBMC3k dataset with test annotations:

        - `.obs['batch']`: 3 batches for integration testing
        - `.obs['cell_type']`: 5 immune cell types for evaluation
        - `.layers['counts']`: Raw count data before normalization
        - `.obsm['X_pca']`: PCA coordinates for downstream analysis
    """
    adata = sc.datasets.pbmc3k()

    # Basic filtering and preprocessing
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    adata.var_names_make_unique()

    # Store raw counts before normalization and add test annotations
    adata.layers["counts"] = adata.X.copy()
    _add_categorical_annotation(adata, "batch", [f"batch_{i}" for i in range(3)])
    _add_categorical_annotation(adata, "cell_type", ["T_cells", "B_cells", "NK_cells", "Monocytes", "Dendritic"])

    # Standard preprocessing
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, batch_key="batch")
    sc.tl.pca(adata, mask_var="highly_variable")

    return adata


def _generate_spatial_coordinates_for_batch(batch_idx: int, n_cells: int, seed: int = 42) -> np.ndarray:
    """Generate synthetic spatial coordinates for a batch of cells.

    Parameters
    ----------
    batch_idx
        Index of the batch (determines spatial offset).
    n_cells
        Number of cells to generate coordinates for.
    seed
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Array of shape (n_cells, 2) with x, y coordinates.
    """
    np.random.seed(seed + batch_idx)  # Different seed per batch

    # Generate circular/radial distribution
    angles = np.random.uniform(0, 2 * np.pi, n_cells)
    radii = np.random.exponential(25, n_cells)

    # Each batch gets a different offset
    offset_x = batch_idx * 50 + np.random.uniform(-10, 10)
    offset_y = batch_idx * 50 + np.random.uniform(-10, 10)

    x_coords = offset_x + radii * np.cos(angles)
    y_coords = offset_y + radii * np.sin(angles)

    return np.column_stack([x_coords, y_coords])


@pytest.fixture
def spatial_data(pbmc_data) -> AnnData:
    """PBMC3k dataset with synthetic spatial coordinates for spatial integration testing.

    Builds on the pbmc_data fixture and adds synthetic spatial coordinates per batch.
    Each batch has its own coordinate system to simulate realistic spatial experiments.

    Parameters
    ----------
    pbmc_data
        Preprocessed PBMC3k dataset from pbmc_data fixture.

    Returns
    -------
    AnnData
        PBMC3k dataset with synthetic spatial information:

        - All features from pbmc_data fixture
        - `.obsm['spatial']`: Synthetic spatial coordinates per batch
    """
    # Use the pbmc_data directly (pytest creates fresh instances per test)
    adata = pbmc_data

    # Generate spatial coordinates for each batch and concatenate
    spatial_coords = []
    for batch_idx, batch_name in enumerate(adata.obs["batch"].cat.categories):
        batch_mask = adata.obs["batch"] == batch_name
        n_cells = int(batch_mask.sum())

        if n_cells > 0:
            batch_coords = _generate_spatial_coordinates_for_batch(batch_idx, n_cells)
            spatial_coords.append((batch_mask, batch_coords))

    # Store all spatial coordinates
    adata.obsm["spatial"] = np.zeros((adata.n_obs, 2))
    for batch_mask, batch_coords in spatial_coords:
        adata.obsm["spatial"][batch_mask] = batch_coords

    return adata
