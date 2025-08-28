# Comparing embeddings for single-cell and spatial data

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/quadbio/scembed/test.yaml?branch=main
[badge-docs]: https://img.shields.io/readthedocs/scembed

Single-cell RNA-sequencing (scRNA-seq) measures gene expression in individual cells and generates large datasets. Typically, these datasets consist of several samples, each corresponding to a combination of covariates (e.g. patient, time point, disease status, technology, etc.). Analyzing these vast datasets (often containing millions of cells for thousands of genes) is facilitated by data integration approaches, which learn lower-dimensional representations that remove the effects of certain unwanted covariates (such as experimental batch, the chip the data was run on, etc).

Here, we use `slurm_sweep` to efficiently parallelize and track different data integration approaches, and we compare their performance in terms of [scIB metrics](https://scib-metrics.readthedocs.io/en/stable/). For each data integration method, we compute a shared latent space, quantify integration performance in terms of batch correction and bio conservation, visualize the latent space with UMAP, store the model and embedding coordinates, and store all relevant data on wandb, so that we can retrieve it after the sweep.

`scembed` consists of shallow wrappers around commonly used integration tools, a class to facilitate scIB comparisons, and another class to retrieve and aggregate sweep results.

### Methods included
- **GPU-based methods**: scVI, scANVI, scPoli, ResolVI, scVIVA
- **CPU-based methods**: Harmony, LIGER, Scanorama, HVG, Pre-computed embeddings

### Evaluation
- **scIB metrics**: Standardized benchmarking for integration quality
- **UMAP visualization**: Visual assessment of integration
- **Artifact tracking**: Models and embeddings stored in wandb

## Outputs

### Per Method
- **Integration embedding**: Stored in wandb as table
- **scIB metrics**: Comprehensive benchmarking scores
- **UMAP plots**: Visualization by cell type and batch
- **Model weights**: For deep learning methods

### Summary Metrics
- **scib_total_score**: Overall integration quality
- **scib_bio_conservation**: Preservation of biological signal
- **scib_batch_correction**: Removal of batch effects

## Getting started

Please refer to the [documentation][],
in particular, the [API documentation][].

## Installation

You need to have Python 3.10 or newer installed on your system.
If you don't have Python installed, we recommend installing [uv][].

There are several alternative options to install scembed:

<!--
1) Install the latest release of `scembed` from [PyPI][]:

```bash
pip install scembed
```
-->

1. Install the latest development version:

```bash
pip install git+https://github.com/quadbio/scembed.git@main
```

**Note**: If you encounter C++ compilation errors (e.g., with `louvain` or `annoy`), install those packages via conda first:
```bash
mamba install louvain python-annoy
```

### Dependency Groups

The package uses optional dependency groups to minimize installation overhead:

- **Base**: Core functionality (scanpy, scib-metrics, wandb)
- **`[cpu]`**: CPU-based methods (e.g. Harmony, LIGER, Scanorama)
- **`[gpu]`**: GPU-based methods (e.g. scVI, scANVI, scPoli)
- **`[fast_metrics]`**: Accelerated evaluation with `faiss` and `RAPIDS`
- **`[all]`**: All optional dependencies

## Release notes

See the [changelog][].

## Contact

For questions and help requests, you can reach out in the [scverse discourse][].
If you found a bug, please use the [issue tracker][].

## Citation

> t.b.a

[uv]: https://github.com/astral-sh/uv
[scverse discourse]: https://discourse.scverse.org/
[issue tracker]: https://github.com/quadbio/scembed/issues
[tests]: https://github.com/quadbio/scembed/actions/workflows/test.yaml
[documentation]: https://scembed.readthedocs.io
[changelog]: https://scembed.readthedocs.io/en/latest/changelog.html
[api documentation]: https://scembed.readthedocs.io/en/latest/api.html
[pypi]: https://pypi.org/project/scembed
