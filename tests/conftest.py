import pytest
import scanpy as sc


@pytest.fixture
def lung_data():
    """Lung atlas dataset fixture.

    Dissociated scRNA-seq dataset, used for demonstration in scIB metrics package.
    32,472 cells × 15,148 genes across multiple batches (`.obs['batch']`), includes cell-type labels (`.obs['cell_type']`).
    Raw counts are in `layers['counts']`.
    """
    adata = sc.read("data/lung_atlas.h5ad", backup_url="https://figshare.com/ndownloader/files/24539942")

    # Simple preprocessing
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="cell_ranger", batch_key="batch")
    sc.tl.pca(adata, n_comps=30, mask_var="highly_variable")

    return adata


@pytest.fixture
def spatial_data():
    """Spatial transcriptomics dataset fixture.

    Spatial data (seqFISH) from Lohoff et al (Nature Biotech 2022), profiling mouse embryogenesis at E8.5.
    51,787 cells × 351 genes across three slices (`.obs['embryo']`), includes cell-type labels (`.obs['celltype_harmonized']`).
    PCA has been pre-computed and is in `.obsm['X_pca']`, spatial coordinates are in `.obsm['spatial']`.
    """
    adata = sc.read("data/spatial_data.h5ad", backup_url="https://figshare.com/ndownloader/files/54145250")

    # Simple preprocessing
    adata.layers["counts"] = adata.X.copy()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    return adata
