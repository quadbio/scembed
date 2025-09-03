import tempfile

import numpy as np
import pytest
import wandb

import scembed

# Global model parameters to avoid duplication
MODEL_PARAMS = {
    "cell_type_key": "cell_type",
    "batch_key": "batch",
    "use_hvg": False,
    "max_epochs": 3,
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
                "extra_params": {},
            },
            {
                "method_name": "scANVI",
                "method_class": scembed.methods.scANVIMethod,
                "embedding_key": "X_scanvi",
                "wandb_embedding_key": "X_scanvi_wandb",
                "extra_params": {"scvi_params": {"max_epochs": 3}},
            },
        ],
    )
    def test_embedding_retrieval(self, pbmc_data, method_config):
        """Test retrieval of embeddings from wandb for different methods.

        Train model, retrieve embedding, log model + embedding to wandb. Then, retrieve both and compare two different embeddings:
        1. The embedding retrieved from wandb.
        2. An embedding obtained by getting the latent space from the fitted model.
        """
        method_name = method_config["method_name"]
        method_class = method_config["method_class"]
        embedding_key = method_config["embedding_key"]
        wandb_embedding_key = method_config["wandb_embedding_key"]
        extra_params = method_config["extra_params"]

        # Step 1: Fit a model, log to wandb
        temp_dir = tempfile.TemporaryDirectory()
        wandb.init(entity="spatial_vi", project="scembed_test_retrival", dir=temp_dir.name)
        run_id = wandb.run.id

        # Initialize the method with common params + method-specific params
        method = method_class(pbmc_data, **MODEL_PARAMS, **extra_params)
        method.fit()
        method.transform()

        model_path = method.save_model(method.models_dir)
        wandb.log_model(str(model_path), name="trained_model")

        emb_path = method.save_embedding(format_type="parquet")
        artifact = wandb.Artifact("embedding", type="dataset")
        artifact.add_file(str(emb_path))
        wandb.log_artifact(artifact)

        wandb.finish()
        temp_dir.cleanup()

        # Step 2: Retrieve artifacts from wandb, get latent space from model
        method = method_class(pbmc_data, **MODEL_PARAMS, **extra_params)

        source = {"entity": "spatial_vi", "project": "scembed_test_retrival", "run_id": run_id}
        method.load_artifact(artifact_type="model", source=source)
        method.load_artifact(artifact_type="embedding", source=source, embedding_key=wandb_embedding_key)

        method.transform()

        # Step 3: make sure the two embeddings are now identical
        assert np.array_equal(method.adata.obsm[wandb_embedding_key], method.adata.obsm[embedding_key]), (
            f"Embeddings are not equal for {method_name}"
        )
