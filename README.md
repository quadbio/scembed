# 🧬 Comparing embeddings for single-cell and spatial data

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]
[![Coverage][badge-coverage]][coverage]
[![Pre-commit.ci][badge-pre-commit]][pre-commit]
[![PyPI][badge-pypi]][pypi]
[![Downloads][badge-downloads]][downloads]
[![Zenodo][badge-zenodo]][zenodo]


[badge-tests]: https://img.shields.io/github/actions/workflow/status/quadbio/scembed/test.yaml?branch=main
[badge-docs]: https://img.shields.io/readthedocs/scembed
[badge-coverage]: https://codecov.io/gh/quadbio/scembed/branch/main/graph/badge.svg
[badge-pre-commit]: https://results.pre-commit.ci/badge/github/quadbio/scembed/main.svg
[badge-pypi]: https://img.shields.io/pypi/v/scembed.svg
[badge-downloads]: https://static.pepy.tech/badge/scembed
[badge-zenodo]: https://zenodo.org/badge/1046168919.svg


Single-cell RNA-sequencing (scRNA-seq) 🧪 measures gene expression in individual cells and generates large datasets. Typically, these datasets consist of several samples, each corresponding to a combination of covariates (e.g. patient, time point, disease status, technology, etc.). Analyzing these vast datasets (often containing millions of cells for thousands of genes) is facilitated by data integration approaches, which learn lower-dimensional representations that remove the effects of certain unwanted covariates (such as experimental batch, the chip the data was run on, etc).

## 🎯 Overview
Here, we use `slurm_sweep` to efficiently parallelize and track different data integration approaches, and we compare their performance in terms of [scIB metrics](https://scib-metrics.readthedocs.io/en/stable/) ([Luecken et al., 2022](https://doi.org/10.1038/s41592-021-01336-8)). For each data integration method, we compute a shared latent space, quantify integration performance in terms of batch correction and bio conservation, visualize the latent space with UMAP, store the model and embedding coordinates, and store all relevant data on wandb, so that we can retrieve it after the sweep.

`scembed` consists of shallow wrappers around commonly used integration tools, a class to facilitate scIB comparisons, and another class to retrieve and aggregate sweep results.


## 🚀 Getting started

Please refer to the [documentation][],
in particular, the [API documentation][].

## 📦 Installation

You need to have Python 3.10 or newer installed on your system.
If you don't have Python installed, we recommend installing [uv][].

There are several alternative options to install scembed:

1. Install the latest release of `scembed` from [PyPI][]:

```bash
pip install scembed
```

2. Install the latest development version:

```bash
pip install git+https://github.com/quadbio/scembed.git@main
```

### 🎯 Dependency Groups

The package uses optional dependency groups to minimize installation overhead:

- **Base**: Core functionality (scanpy, scib-metrics, wandb)
- **`[cpu]`**: CPU-based methods (e.g. Harmony, LIGER, Scanorama)
- **`[gpu]`**: GPU-based methods (e.g. scVI, scANVI, scPoli)
- **`[fast_metrics]`**: Accelerated evaluation with `faiss` and `RAPIDS` ⚡
- **`[all]`**: All optional dependencies

**⚠️ Note**: If you encounter C++ compilation errors (e.g., with `louvain` or `annoy`), install those packages via conda/mamba first:
```bash
mamba install louvain python-annoy
```

## 📝 Release notes

See the [changelog][].

## 💬 Contact

For questions and help requests, you can reach out in the [scverse discourse][].
If you found a bug, please use the [issue tracker][].

## 📖 Citation

Please use our [zenodo][] entry to cite this software.

[uv]: https://github.com/astral-sh/uv
[scverse discourse]: https://discourse.scverse.org/
[issue tracker]: https://github.com/quadbio/scembed/issues
[tests]: https://github.com/quadbio/scembed/actions/workflows/test.yaml
[documentation]: https://scembed.readthedocs.io
[changelog]: https://scembed.readthedocs.io/en/latest/changelog.html
[api documentation]: https://scembed.readthedocs.io/en/latest/api.html
[pypi]: https://pypi.org/project/scembed

[coverage]: https://codecov.io/gh/quadbio/scembed
[pre-commit]: https://results.pre-commit.ci/latest/github/quadbio/scembed/main
[downloads]: https://pepy.tech/project/scembed
[zenodo]: https://doi.org/10.5281/zenodo.16982001
