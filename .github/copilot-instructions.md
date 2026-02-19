# Copilot Instructions for scembed

## Project Overview

**scembed** is a Python package for comparing single-cell data integration
methods using [scIB metrics](https://scib-metrics.readthedocs.io/). It wraps
common integration tools, facilitates benchmarking with scIB, and aggregates
results from W&B sweeps run via [slurm_sweep](https://github.com/quadbio/slurm_sweep).

### Domain Context
- **Integration methods**: Remove batch effects while preserving biological
  variation. Examples: Harmony (CPU), scVI/scANVI/scPoli (GPU, from scvi-tools).
- **scIB metrics**: Benchmark integration quality via batch correction (iLISI,
  KBET) and bio conservation (ARI, NMI, silhouette).
- **W&B (Weights & Biases)**: Experiment tracking. We log configs, metrics,
  embeddings, and models as artifacts.

### Key Dependencies
- **Core**: scanpy, anndata, scib-metrics, wandb
- **Integration methods**: Optional extras `[cpu]`, `[gpu]`, `[fast-metrics]`
- **scvi-tools**: PyTorch-based methods (scVI, scANVI, scPoli)

## Architecture

### Core Components
1. **`src/scembed/methods/`**: Wrappers for integration methods
   - Base class: `BaseIntegrationMethod` (abstract)
   - CPU methods: Harmony, LIGER, Scanorama, precomputed PCA
   - GPU methods: scVI, scANVI, scPoli, ResolVI
2. **`src/scembed/evaluation.py`**: `IntegrationEvaluator` for scIB benchmarking
3. **`src/scembed/aggregation.py`**: `scIBAggregator` for W&B sweep result aggregation
4. **`src/scembed/utils.py`**: Utilities (subsampling, artifact download, embedding I/O)

## Project-Specific Patterns

### Working with Integration Methods
```python
method = HarmonyMethod(
    adata=adata,
    batch_key="batch",
    cell_type_key="celltype",
    output_dir=Path("outputs"),
)
method.integrate()  # creates .obsm['X_harmony']
method.save_embedding()
```

### W&B Integration
- Methods can log to W&B via `wandb.init()` (typically in training scripts)
- Artifacts logged: trained models, embeddings
- `scIBAggregator` fetches and processes sweep results from W&B API
- Handle W&B SDK version differences gracefully (see helper methods in `aggregation.py`)

## Common Gotchas

1. **Optional dependencies**: CPU/GPU methods require respective extras. Check with `check_deps()` from `scembed.check`.
2. **W&B API changes**: SDK behavior varies across versions. Use helper methods for parsing run data (see `aggregation.py`).

## Related Resources

- **scIB metrics docs**: https://scib-metrics.readthedocs.io/
- **scvi-tools docs**: https://docs.scvi-tools.org/
- **slurm_sweep**: https://github.com/quadbio/slurm_sweep
