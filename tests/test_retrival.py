import tempfile

import numpy as np
import wandb

import scembed


class TestRetrieval:
    """Test retrieval of artifacts from wandb.

    In particular, make sure that retrieving embedding coordinates from fitted models logged to wandb
    is consistent with the original stored embeddings.
    """

    def test_scvi_embedding_retrieval(self, pbmc_data):
        """Test retrieval of scVI embeddings from wandb.

        Train scVI model, retrive embedding, log model + embedding to wandb. Then, retrive both and compare two different embeddings:
        1. The embedding retrieved from wandb.
        2. An embedding obtained by getting the latent space from the fitted model.
        """

        # Step 1: Fit a model, log to wandb
        temp_dir = tempfile.TemporaryDirectory()
        wandb.init(entity="spatial_vi", project="scembed_test_retrival", dir=temp_dir.name)
        run_id = wandb.run.id

        method = scembed.methods.scVIMethod(pbmc_data, use_hvg=False, max_epochs=5)
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
        method = scembed.methods.scVIMethod(pbmc_data, use_hvg=False)
        source = {"entity": "spatial_vi", "project": "scembed_test_retrival", "run_id": run_id}
        method.load_artifact(artifact_type="model", source=source)
        method.load_artifact(artifact_type="embedding", source=source, embedding_key="X_scvi_wandb")

        method.transform()

        # Step 3: make sure the two embeddings are now identical
        assert np.array_equal(method.adata.obsm["X_scvi_wandb"], method.adata.obsm["X_scvi"]), (
            "Embeddings are not equal"
        )
