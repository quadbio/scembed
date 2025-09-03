"""Base class for integration methods."""

import gzip
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy.sparse import issparse

from scembed.logging import logger
from scembed.utils import _download_artifact_by_run_id, load_embedding


class BaseIntegrationMethod(ABC):
    """Abstract base class for single-cell integration methods."""

    def __init__(
        self,
        adata: ad.AnnData,
        output_dir: str | Path | None = None,
        validate_spatial: bool = False,
        batch_key: str = "batch",
        cell_type_key: str = "cell_type",
        hvg_key: str = "highly_variable",
        use_hvg: bool = True,
        counts_layer: str = "counts",
        spatial_key: str = "spatial",
        pca_key: str = "X_pca",
        unlabeled_category: str = "unknown",
        unlabeled_color: str = "#8f8f8f",
        **kwargs,
    ):
        """
        Initialize the integration method.

        Parameters
        ----------
        adata
            Annotated data object to validate and store.
        output_dir
            Directory for saving outputs. If None, creates a temporary directory.
        validate_spatial
            Whether to validate spatial data requirements.
        batch_key
            Key in adata.obs for batch information.
        cell_type_key
            Key in adata.obs for cell type information.
        hvg_key
            Key in adata.var for highly variable genes.
        use_hvg
            Whether to use highly variable genes for integration.
        counts_layer
            Key in adata.layers for count data.
        spatial_key
            Key in adata.obsm for spatial coordinates.
        pca_key
            Key in adata.obsm for PCA embedding.
        unlabeled_category
            Category name for unlabeled cells in label-based methods.
        unlabeled_color
            Color for unlabeled cells in label-based methods.
        **kwargs
            Method-specific parameters.
        """
        self.name = self.__class__.__name__.replace("Method", "")
        self.params = kwargs
        self.is_fitted = False
        self.embedding_key = f"X_{self.name.lower()}"
        self.model = None  # For methods that have trainable models

        # Data keys - configurable for different datasets
        self.batch_key = batch_key
        self.cell_type_key = cell_type_key
        self.hvg_key = hvg_key
        self.use_hvg = use_hvg
        self.counts_layer = counts_layer
        self.spatial_key = spatial_key
        self.pca_key = pca_key
        self.unlabeled_category = unlabeled_category
        self.unlabeled_color = unlabeled_color

        # Validate and store the data
        adata_work = self.validate_adata(adata.copy())
        if validate_spatial:
            self.validate_spatial_adata(adata_work)
        self.adata = adata_work

        # Setup state
        self.setup_state = {
            "is_setup": False,
            "adata_prepared": None,
        }

        # Setup output directories
        self._temp_dir = None  # Store TemporaryDirectory object to prevent premature deletion
        if output_dir is None:
            self._temp_dir = TemporaryDirectory()
            output_dir = Path(self._temp_dir.name)
        elif isinstance(output_dir, str):
            output_dir = Path(output_dir)

        # Create sub-directories
        self.output_dir = output_dir
        self.models_dir = output_dir / "models"
        self.embedding_dir = output_dir / "embeddings"

        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Initialized %s method, saving outputs to '%s'.", self.name, self.output_dir)

    def validate_adata(self, adata: ad.AnnData) -> ad.AnnData:
        """
        Validate the AnnData object has required keys and structure.

        Parameters
        ----------
        adata
            Annotated data object to validate.

        Raises
        ------
        ValueError
            If required keys are missing or data is malformed.
        """
        # Check required observation keys
        if self.batch_key not in adata.obs.columns:
            raise ValueError(f"Batch key '{self.batch_key}' not found in adata.obs")

        # Validate and process cell type key
        self._validate_cell_type_key(adata)

        # Check if counts layer exists (if not using X directly)
        if self.counts_layer != "X" and self.counts_layer not in adata.layers:
            logger.warning("Counts layer '%s' not found in adata.layers", self.counts_layer)
        else:
            count_data = adata.layers[self.counts_layer] if self.counts_layer != "X" else adata.X
            is_integer = np.all((count_data.data if issparse(count_data) else count_data) % 1 == 0)

            if not is_integer:
                logger.warning("Counts layer '%s' contains non-integer values", self.counts_layer)

        # Check for highly variable genes (most methods will need this)
        if self.hvg_key not in adata.var.columns and self.use_hvg:
            logger.warning("HVG key '%s' not found in adata.var. Using all genes.", self.hvg_key)
            self.use_hvg = False

        # Check for PCA embedding (some methods will need this)
        if self.pca_key not in adata.obsm:
            logger.warning(
                "PCA embedding '%s' not found in adata.obsm. Methods like Harmony require this.", self.pca_key
            )

        logger.info("Data validation passed for %s method.", self.name)

        return adata

    def _validate_cell_type_key(self, adata: ad.AnnData) -> None:
        """Validate and process cell type key with unlabeled category handling."""
        if self.cell_type_key not in adata.obs.columns:
            logger.warning(
                "Cell type key '%s' not found in adata.obs. Label-based methods like scANVI require this.",
                self.cell_type_key,
            )
            return

        # Convert to categorical if needed
        cell_type_col = adata.obs[self.cell_type_key]
        if not isinstance(cell_type_col.dtype, pd.CategoricalDtype):
            logger.debug("Converting cell type key '%s' to categorical", self.cell_type_key)
            adata.obs[self.cell_type_key] = cell_type_col.astype("category")
            cell_type_col = adata.obs[self.cell_type_key]

        # Check if unlabeled category exists
        has_unlabeled = self.unlabeled_category in cell_type_col.cat.categories
        if has_unlabeled:
            n_unlabeled = cell_type_col.value_counts().get(self.unlabeled_category, 0)
            logger.debug(
                "Unlabeled category '%s' with %d cells found in cell type key '%s'",
                self.unlabeled_category,
                n_unlabeled,
                self.cell_type_key,
            )
        else:
            logger.warning(
                "Unlabeled category '%s' not found in cell type key '%s'",
                self.unlabeled_category,
                self.cell_type_key,
            )

        # Handle missing values by converting to unlabeled category
        n_missing = cell_type_col.isna().sum()
        if n_missing > 0:
            logger.warning(
                "Found %d missing values in cell type key '%s'. Converting to '%s'",
                n_missing,
                self.cell_type_key,
                self.unlabeled_category,
            )

            if not has_unlabeled:
                self._add_unlabeled_category(adata)
                # Update our reference to the modified categorical
                cell_type_col = adata.obs[self.cell_type_key]

            adata.obs[self.cell_type_key] = cell_type_col.fillna(self.unlabeled_category)

    def _add_unlabeled_category(self, adata: ad.AnnData) -> None:
        """Add unlabeled category to cell type key and update colors if they exist."""
        # Preserve existing color mapping
        color_key = f"{self.cell_type_key}_colors"
        cmap = None
        if color_key in adata.uns:
            cmap = dict(
                zip(
                    adata.obs[self.cell_type_key].cat.categories,
                    adata.uns[color_key],
                    strict=True,
                )
            )

        # Add unlabeled category
        adata.obs[self.cell_type_key] = adata.obs[self.cell_type_key].cat.add_categories(self.unlabeled_category)

        # Update color mapping if it existed
        if cmap is not None:
            cmap[self.unlabeled_category] = self.unlabeled_color
            adata.uns[color_key] = [cmap[cat] for cat in adata.obs[self.cell_type_key].cat.categories]

    def validate_spatial_adata(self, adata: ad.AnnData) -> None:
        """
        Validate spatial-specific data requirements.

        Parameters
        ----------
        adata
            Annotated data object to validate.

        Raises
        ------
        ValueError
            If required spatial keys are missing or data is malformed.
        """
        # Check for spatial coordinates using the configured spatial_key
        if self.spatial_key not in adata.obsm:
            raise ValueError(f"Spatial coordinates not found. Expected '{self.spatial_key}' in adata.obsm")

        # Check spatial coordinates format
        spatial_coords = adata.obsm[self.spatial_key]
        if spatial_coords.shape[1] != 2:
            raise ValueError("Spatial coordinates must have 2 dimensions (x, y)")

        # Check for precomputed spatial neighbors (methods will compute these if missing)
        spatial_keys = [
            key
            for key in adata.obsm.keys() | adata.obsp.keys()
            if any(prefix in key.lower() for prefix in ["spatial", "index_neighbor", "distance_neighbor"])
        ]

        if not spatial_keys:
            logger.warning("No precomputed spatial neighbors found. Spatial methods will compute these during setup.")

        logger.info("Spatial data validation passed for %s method.", self.name)

    @abstractmethod
    def fit(self) -> None:
        """Fit the integration method to the data.

        Uses self.adata which was validated during initialization.
        """
        raise NotImplementedError

    @abstractmethod
    def transform(self) -> None:
        """Transform the data and add embedding to obsm.

        Uses self.adata which was validated during initialization.
        Modifies self.adata in place by adding embedding to .obsm[self.embedding_key].
        """
        raise NotImplementedError

    def fit_transform(self) -> None:
        """
        Fit the method and transform the data.

        Modifies self.adata in place.
        """
        self.fit()
        self.transform()

    def setup(self, force_recompute: bool = False) -> None:
        """
        Setup data preprocessing for the integration method.

        Prepares the data for method-specific training/inference and stores
        the result in self.setup_state["adata_prepared"].

        Parameters
        ----------
        force_recompute
            Whether to force recomputation of preprocessing steps.
        """
        # Skip if already setup and not forcing recompute
        if self.setup_state["is_setup"] and not force_recompute:
            return

        # Default implementation: just use the original data
        self.setup_state["adata_prepared"] = self.adata.copy()
        self.setup_state["is_setup"] = True

        logger.info("Data setup completed for %s method", self.name)

    def load_artifact(
        self,
        source: str | Path | dict,
        artifact_type: Literal["model", "embedding"] = "model",
        embedding_key: str | None = None,
        **kwargs,
    ) -> None:
        """
        Load a pre-trained model or embedding from various sources.

        Parameters
        ----------
        source
            Source of the artifact. Can be:
            - str/Path: Local path to model directory or embedding file
            - dict: WandB parameters with keys 'run_id', 'entity', 'project'
        artifact_type
            Type of artifact to load: 'model' or 'embedding'.
        embedding_key
            Key to store embedding in adata.obsm. If None, uses self.embedding_key.
            Only used when artifact_type='embedding'.
        **kwargs
            Additional arguments passed to loading functions.
        """
        if isinstance(source, dict):
            # WandB loading - download first, then load with appropriate method
            if artifact_type == "model":
                artifact_name = "trained_model"
                download_dir = self.models_dir
            else:  # artifact_type == "embedding"
                artifact_name = "embedding"
                download_dir = self.embedding_dir

            downloaded_path = self._load_from_wandb(artifact_name=artifact_name, download_dir=download_dir, **source)

            if downloaded_path is None:
                raise ValueError(
                    f"Could not download artifact '{artifact_name}' from run {source.get('run_id', 'unknown')}"
                )

            # Call appropriate loading method based on artifact type
            if artifact_type == "model":
                self._load_from_disk(downloaded_path, **kwargs)
            else:  # artifact_type == "embedding"
                self._load_embedding_from_disk(downloaded_path, embedding_key=embedding_key, **kwargs)
        else:
            # Local path loading
            if artifact_type == "model":
                self._load_from_disk(Path(source), **kwargs)
            else:  # artifact_type == "embedding"
                self._load_embedding_from_disk(Path(source), embedding_key=embedding_key, **kwargs)

    def _load_from_wandb(
        self,
        run_id: str,
        entity: str,
        project: str,
        artifact_name: str = "trained_model",
        download_dir: Path | None = None,
    ) -> Path:
        """
        Download artifact from WandB and return the path.

        Parameters
        ----------
        run_id
            WandB run ID.
        entity
            WandB entity.
        project
            WandB project.
        artifact_name
            Name of the artifact to download.
        download_dir
            Directory to download to. If None, uses models_dir.

        Returns
        -------
        Path
            Path to the downloaded artifact.
        """
        if download_dir is None:
            download_dir = self.models_dir

        downloaded_path = _download_artifact_by_run_id(
            run_id=run_id,
            entity=entity,
            project=project,
            artifact_name=artifact_name,
            download_dir=download_dir,
        )

        if downloaded_path is None:
            raise ValueError(f"Could not download artifact '{artifact_name}' from run {run_id}")

        return downloaded_path

    def _load_from_disk(self, model_path: Path, **kwargs) -> None:
        """
        Load model from local disk.

        Parameters
        ----------
        model_path
            Path to the model directory.
        **kwargs
            Additional arguments for model loading.

        Raises
        ------
        NotImplementedError
            If method doesn't support model loading.
        """
        raise NotImplementedError

    def _load_embedding_from_disk(self, embedding_path: Path, embedding_key: str | None = None, **kwargs) -> None:
        """
        Load embedding from local disk.

        Parameters
        ----------
        embedding_path
            Path to the embedding file or directory containing embedding files.
        embedding_key
            Key to store embedding in adata.obsm. If None, uses self.embedding_key.
        **kwargs
            Additional arguments for embedding loading (unused, for compatibility).
        """
        _ = kwargs  # Silence unused parameter warning

        # Use provided embedding key or default to method's embedding key
        target_key = embedding_key if embedding_key is not None else self.embedding_key

        # If path is a directory (e.g., downloaded WandB artifact), find the embedding file
        if embedding_path.is_dir():
            # Look for common embedding file patterns
            embedding_files = []
            for pattern in ["*.parquet", "*.pkl.gz", "*.h5"]:
                embedding_files.extend(embedding_path.glob(pattern))

            if not embedding_files:
                raise ValueError(f"No embedding files found in directory: {embedding_path}")
            elif len(embedding_files) > 1:
                logger.warning("Multiple embedding files found, using the first one: %s", embedding_files[0])

            embedding_file = embedding_files[0]
        else:
            embedding_file = embedding_path

        # Load embedding using utility function
        embedding_df = load_embedding(embedding_file)

        # Store in adata.obsm
        self.adata.obsm[target_key] = embedding_df.values

        logger.info("Loaded %s embedding from '%s' into key '%s'", self.name, embedding_file, target_key)

    def _load_scvi_model(self, model_path: Path, model_class_path: str, **kwargs) -> None:
        """
        Helper method for loading scVI-tools based models.

        Parameters
        ----------
        model_path
            Path to the model directory.
        model_class_path
            Dot-separated path to the model class (e.g., 'scvi.model.SCVI').
        **kwargs
            Additional arguments for model setup
        """
        # Setup data first
        self.setup(**kwargs)

        # Dynamically import the model class
        module_name, class_name = model_class_path.rsplit(".", 1)
        module = __import__(module_name, fromlist=[class_name])
        model_class = getattr(module, class_name)

        # Load model
        self.model = model_class.load(str(model_path), adata=self.setup_state["adata_prepared"])

        self.is_fitted = True

    def save_model(self, path: Path) -> Path | None:
        """
        Save the trained model (for deep learning methods).

        Parameters
        ----------
        path
            Directory to save the model.

        Returns
        -------
        Optional[Path]
            Path to saved model file, None if method doesn't support saving.
        """
        # Default implementation - subclasses can override
        _ = path  # Silence unused parameter warning
        return None

    def save_embedding(
        self,
        format_type: Literal["parquet", "pickle", "h5"] = "parquet",
        filename: str | None = None,
        compression: bool = True,
    ) -> Path:
        """
        Save embedding to file with preserved cell names as index.

        Parameters
        ----------
        format_type
            Format to save embedding in. Options: 'parquet', 'pickle', or 'h5'.
        filename
            Custom filename (without extension). If None, uses "embedding".
        compression
            Whether to use compression (gzip for all formats).

        Returns
        -------
        Path
            Path to the saved embedding file.

        Raises
        ------
        ValueError
            If method is not fitted or embedding key not found in adata.obsm.
        """
        if not self.is_fitted:
            raise ValueError("Method must be fitted before saving embedding")
        if self.embedding_key not in self.adata.obsm:
            raise ValueError(f"Embedding key '{self.embedding_key}' not found in adata.obsm")

        filename = filename or "embedding"

        # Create DataFrame (shared for parquet/pickle)
        emb_df = pd.DataFrame(
            data=self.adata.obsm[self.embedding_key],
            index=self.adata.obs_names,
            columns=[f"dim_{i}" for i in range(self.adata.obsm[self.embedding_key].shape[1])],
        )

        if format_type == "parquet":
            file_path = self.embedding_dir / f"{filename}.parquet"
            emb_df.to_parquet(file_path, compression="gzip" if compression else None)

        elif format_type == "pickle":
            file_path = self.embedding_dir / f"{filename}.pkl.gz"
            with gzip.open(file_path, "wb") as f:
                pickle.dump(emb_df, f)

        elif format_type == "h5":
            file_path = self.embedding_dir / f"{filename}.h5"
            with h5py.File(file_path, "w") as hf:
                hf.create_dataset(
                    "embedding", data=self.adata.obsm[self.embedding_key], compression="gzip" if compression else None
                )
                hf.create_dataset("cell_names", data=[n.encode() for n in self.adata.obs_names])
                hf.create_dataset(
                    "dim_names", data=[f"dim_{i}".encode() for i in range(self.adata.obsm[self.embedding_key].shape[1])]
                )
        else:
            raise ValueError(f"Unsupported format_type: {format_type}. Choose from 'parquet', 'pickle', 'h5'")

        logger.info("Saved %s embedding to '%s'", self.name, file_path)
        return file_path

    def _prepare_hvg(self) -> ad.AnnData:
        """
        Prepare AnnData object with HVG subsetting if needed.

        Returns
        -------
        AnnData
            Prepared data with HVG subsetting applied if use_hvg=True.
        """
        if self.use_hvg:
            if self.hvg_key not in self.adata.var.columns:
                raise ValueError(f"HVG key '{self.hvg_key}' not found but use_hvg=True")
            return self.adata[:, self.adata.var[self.hvg_key]].copy()
        else:
            return self.adata.copy()

    def _filter_none_params(self, params: dict) -> dict:
        """Filter out None values to allow library defaults."""
        return {k: v for k, v in params.items() if v is not None}

    def get_model_info(self) -> dict[str, Any]:
        """
        Get information about the fitted model.

        Returns
        -------
        Dict[str, Any]
            Dictionary with model information.
        """
        return {
            "method": self.name,
            "params": self.params,
            "is_fitted": self.is_fitted,
            "embedding_key": self.embedding_key,
        }

    def __repr__(self) -> str:
        """String representation of the method."""
        params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        status = "fitted" if self.is_fitted else "not fitted"

        # Count HVGs
        if self.use_hvg:
            n_hvgs = self.adata.var[self.hvg_key].sum()
            hvg_info = f"{n_hvgs:,} HVGs"
        else:
            hvg_info = "not using HVGs"

        data_info = f"{self.adata.n_obs:,} cells × {self.adata.n_vars:,} genes ({hvg_info})"
        return f"{self.__class__.__name__}({params_str}) [{status}, {data_info}]"
