# Copilot Instructions for scembed

## Important Notes
- Avoid drafting summary documents or endless markdown files. Just summarize in chat what you did, why, and any open questions.
- Don't update Jupyter notebooks - those are managed manually.
- When running terminal commands, activate the appropriate environment first (use `mamba activate slurm_sweep`).
- Rather than making assumptions, ask for clarification when uncertain.
- **GitHub workflows**: Use GitHub CLI (`gh`) when possible. For GitHub MCP server tools, ensure Docker Desktop is running first (`open -a "Docker Desktop"`).


## Project Overview

**scembed** is a Python package for comparing single-cell data integration methods using [scIB metrics](https://scib-metrics.readthedocs.io/). It wraps common integration tools, facilitates benchmarking with scIB, and aggregates results from W&B sweeps run via [slurm_sweep](https://github.com/quadbio/slurm_sweep).

### Domain Context (Brief)
- **AnnData**: Standard single-cell data structure (scanpy ecosystem). Contains `.X` (expression matrix), `.obs` (cell metadata), `.var` (gene metadata), `.layers` (alternative data representations).
- **Integration methods**: Remove batch effects while preserving biological variation. Examples: Harmony (CPU), scVI/scANVI/scPoli (GPU, from scvi-tools).
- **scIB metrics**: Benchmark integration quality via batch correction (iLISI, KBET) and bio conservation (ARI, NMI, silhouette).
- **W&B (Weights & Biases)**: Experiment tracking. We log configs, metrics, embeddings, and models as artifacts.

### Key Dependencies
- **Core**: scanpy, anndata, scib-metrics, wandb
- **Integration methods**: Optional extras `[cpu]`, `[gpu]`, `[fast-metrics]`
- **scvi-tools**: PyTorch-based methods (scVI, scANVI, scPoli)

## Architecture & Code Organization

### Module Structure (follows scverse conventions)
- Use `AnnData` objects as primary data structure
- Type annotations use modern syntax: `str | None` instead of `Optional[str]`
- Supports Python 3.11, 3.12, 3.13 (see `pyproject.toml`)
- Avoid local imports unless necessary for circular import resolution

### Core Components
1. **`src/scembed/methods/`**: Wrappers for integration methods
   - Base class: `BaseIntegrationMethod` (abstract)
   - CPU methods: Harmony, LIGER, Scanorama, precomputed PCA
   - GPU methods: scVI, scANVI, scPoli, ResolVI
2. **`src/scembed/evaluation.py`**: `IntegrationEvaluator` for scIB benchmarking
3. **`src/scembed/aggregation.py`**: `scIBAggregator` for W&B sweep result aggregation
4. **`src/scembed/utils.py`**: Utilities (subsampling, artifact download, embedding I/O)

## Development Workflow

### Environment Management (Hatch-based)
```bash
# Testing - NEVER use pytest directly
hatch test                    # test with highest Python version
hatch test --all              # test all Python 3.11 & 3.13

# Documentation
hatch run docs:build          # build Sphinx docs
hatch run docs:open           # open in browser
hatch run docs:clean          # clean build artifacts

# Environment inspection
hatch env show                # list environments
hatch env find hatch-test     # find test environment paths
```

### Testing Strategy
- Test matrix defined in `[[tool.hatch.envs.hatch-test.matrix]]` in `pyproject.toml`
- CI extracts test config from pyproject.toml (`.github/workflows/test.yaml`)
- Tests live in `tests/`, fixtures in `tests/conftest.py`
- **Always run tests via `hatch test`**, NOT standalone pytest
- Optional dependencies tested via `features = ["test"]` which includes `[cpu]` and `[gpu]`

### Code Quality Tools
- **Ruff**: Linting and formatting (120 char line length)
- **Biome**: JSON/JSONC formatting with trailing commas
- **Pre-commit**: Auto-runs ruff, biome. Install with `pre-commit install`
- Use `git pull --rebase` if pre-commit.ci commits to your branch

## Documentation Conventions

### Docstring Style (NumPy format via Napoleon)
```python
def example_function(
    adata: ad.AnnData,
    *,  # keyword-only marker
    layer_key: str | None = None,
    n_neighbors: int = 15,
) -> ad.AnnData:
    """Short one-line description.

    Extended description if needed.

    Parameters
    ----------
    adata
        AnnData object with preprocessed data.
    layer_key
        Key in .layers to use (if None, uses .X).
    n_neighbors
        Number of neighbors for kNN graph.

    Returns
    -------
    AnnData with integration results in .obsm['X_integrated'].
    """
```

### Sphinx & Documentation
- API docs auto-generated from `docs/api.md` using `autosummary`
- Notebooks in `docs/notebooks/` rendered via myst-nb (`.ipynb` only)
- Add external packages to `intersphinx_mapping` in `docs/conf.py`
- See `docs/contributing.md` for detailed documentation guidelines

## Key Configuration Files

### `pyproject.toml`
- **Build**: `hatchling` with `hatch-vcs` for git-based versioning
- **Dependencies**: Organized as base + optional extras (`[cpu]`, `[gpu]`, `[fast-metrics]`, `[all]`)
- **Ruff**: 120 char line length, NumPy docstring convention
- **Test matrix**: Python 3.11 & 3.13

### Version Management
- Version from git tags via `hatch-vcs`
- Release: Create GitHub release with tag `vX.X.X`
- Follows **Semantic Versioning**

## Project-Specific Patterns

### Working with Integration Methods
```python
# All methods follow this pattern:
method = HarmonyMethod(
    adata=adata,
    batch_key="batch",
    cell_type_key="celltype",
    output_dir=Path("outputs"),
)
method.integrate()  # creates .obsm['X_harmony']
method.save_embedding()  # saves to output_dir/embeddings/
```

### W&B Integration
- Methods can log to W&B via `wandb.init()` (typically in training scripts)
- Artifacts logged: trained models, embeddings
- `scIBAggregator` fetches and processes sweep results from W&B API
- Handle W&B SDK version differences gracefully (see helper methods in `aggregation.py`)

### AnnData Conventions
- Check matrix format: `adata.X` may be sparse or dense
- Use `adata.layers[key]` for alternative representations
- Integration results go in `adata.obsm['X_<method>']`
- Save/load with `.h5ad` (via `sc.read/write`) or `.parquet` for embeddings

### Testing with AnnData
```python
# From conftest.py - example fixture pattern
@pytest.fixture
def adata():
    """Small test AnnData object."""
    adata = ad.AnnData(
        X=np.random.randn(100, 50).astype(np.float32),
        obs=pd.DataFrame({"batch": ["A", "B"] * 50, "celltype": ["T", "B"] * 50}),
    )
    sc.pp.pca(adata, n_comps=10)
    return adata
```

## Common Gotchas

1. **Hatch for testing**: Always use `hatch test`, never standalone `pytest`. CI matches hatch test matrix.
2. **Sparse matrices**: Check `scipy.sparse.issparse(adata.X)` before operations. Use `.X.toarray()` or `adata.layers['counts']` as needed.
3. **Optional dependencies**: CPU/GPU methods require respective extras. Check with `check_deps()` from `scembed.check`.
4. **W&B API changes**: SDK behavior varies across versions. Use helper methods for parsing run data (see `aggregation.py`).
5. **Pre-commit conflicts**: Use `git pull --rebase` to integrate pre-commit.ci fixes.
6. **Line length**: Ruff set to 120 chars, but keep docstrings readable (~80 chars per line).

## Related Resources

- **Contributing guide**: `docs/contributing.md`
- **scIB metrics docs**: https://scib-metrics.readthedocs.io/
- **scvi-tools docs**: https://docs.scvi-tools.org/
- **scanpy tutorials**: https://scanpy.readthedocs.io/
- **slurm_sweep**: https://github.com/quadbio/slurm_sweep
