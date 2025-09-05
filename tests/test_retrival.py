import tempfile

import numpy as np
import pytest
import wandb

import scembed

# Global model parameters to avoid duplication
MODEL_PARAMS = {
    "cell_type_key": "cell_type",
    "batch_key": "batch",
    "use_hvg": True,
    "max_epochs": 1,
    "spatial_key": "spatial",
    "counts_layer": "counts",
}


class TestRetrieval:
    """Test retrieval of artifacts from wandb.

    In particular, make sure that retrieving embedding coordinates from fitted models logged to wandb
    is consistent with the original stored embeddings.
    """

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "method_config",
        [
            {
                "method_name": "scVI",
                "method_class": scembed.methods.scVIMethod,
                "embedding_key": "X_scvi",
                "wandb_embedding_key": "X_scvi_wandb",
                "extra_params": {
                    "n_latent": 5,
                    "n_hidden": 32,
                },
            },
            {
                "method_name": "scANVI",
                "method_class": scembed.methods.scANVIMethod,
                "embedding_key": "X_scanvi",
                "wandb_embedding_key": "X_scanvi_wandb",
                "extra_params": {"scvi_params": {"max_epochs": 1, "n_latent": 5, "n_hidden": 32}},
            },
            {
                "method_name": "scPoli",
                "method_class": scembed.methods.scPoliMethod,
                "embedding_key": "X_scpoli",
                "wandb_embedding_key": "X_scpoli_wandb",
                "requires_scarches": True,
                "extra_params": {
                    "latent_dim": 5,
                    "hidden_layer_sizes": [32],
                    "pretraining_epochs": 1,
                    "n_epochs": 2,
                    "unlabeled_prototype_training": False,
                },
            },
            {
                "method_name": "ResolVI",
                "method_class": scembed.methods.ResolVIMethod,
                "embedding_key": "X_resolvi",
                "wandb_embedding_key": "X_resolvi_wandb",
                "extra_params": {
                    "n_latent": 5,
                    "n_hidden": 32,
                    "n_neighbors": 10,
                    "downsample_counts": False,
                },
                "fixture": "spatial_data",
            },
            {
                "method_name": "scVIVA",
                "method_class": scembed.methods.scVIVAMethod,
                "embedding_key": "X_scviva",
                "wandb_embedding_key": "X_scviva_wandb",
                "extra_params": {
                    "n_latent": 5,
                    "k_nn": 10,
                    "embedding_method": "scvi",
                    "scvi_params": {"max_epochs": 1, "n_latent": 5, "n_hidden": 32},
                },
                "fixture": "spatial_data",
            },
        ],
    )
    def test_embedding_retrieval(self, pbmc_data, spatial_data, method_config):
        """Test retrieval of embeddings from wandb for different methods.

        Train model, retrieve embedding, log model + embedding to wandb. Then, retrieve both and compare two different embeddings:
        1. The embedding retrieved from wandb.
        2. An embedding obtained by getting the latent space from the fitted model.
        """
        # Skip test if method requires scarches but it's not available
        if method_config.get("requires_scarches", False):
            pytest.importorskip("scarches")  # has an issue with AnnData >=0.12

        method_name = method_config["method_name"]
        method_class = method_config["method_class"]
        embedding_key = method_config["embedding_key"]
        wandb_embedding_key = method_config["wandb_embedding_key"]
        extra_params = method_config["extra_params"]

        # Choose the appropriate fixture based on method requirements
        fixture_name = method_config.get("fixture", "pbmc_data")
        if fixture_name == "spatial_data":
            adata = spatial_data
        else:
            adata = pbmc_data

        # Step 1: Fit a model, log to wandb
        temp_dir = tempfile.TemporaryDirectory()
        wandb.init(entity="spatial_vi", project="scembed_test_retrival", dir=temp_dir.name)
        run_id = wandb.run.id

        # Initialize the method with common params + method-specific params
        method_original = method_class(adata, **MODEL_PARAMS, **extra_params)
        method_original.fit()
        method_original.transform()

        model_path = method_original.save_model(method_original.models_dir)
        wandb.log_model(str(model_path), name="trained_model")

        emb_path = method_original.save_embedding(format_type="parquet")
        artifact = wandb.Artifact("embedding", type="dataset")
        artifact.add_file(str(emb_path))
        wandb.log_artifact(artifact)

        wandb.finish()
        temp_dir.cleanup()

        # Step 2: Retrieve artifacts from wandb, get latent space from model
        method_wandb = method_class(adata, **MODEL_PARAMS, **extra_params)

        # We need to provide the original scVI embedding for scVIVA
        if method_name == "scVIVA":
            expression_embedding_key = "X_scvi"
            method_wandb.adata.obsm[expression_embedding_key] = method_original.embedding_model.adata.obsm[
                expression_embedding_key
            ]

        source = {"entity": "spatial_vi", "project": "scembed_test_retrival", "run_id": run_id}
        method_wandb.load_artifact(artifact_type="model", source=source)
        method_wandb.load_artifact(artifact_type="embedding", source=source, embedding_key=wandb_embedding_key)

        method_wandb.transform()

        # Step 3: make sure the two embeddings are now identical
        assert np.array_equal(method_wandb.adata.obsm[embedding_key], method_wandb.adata.obsm[wandb_embedding_key]), (
            f"Embeddings are not equal for {method_name}"
        )
