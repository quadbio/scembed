# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [Unreleased]

### Added

- `SIMVIMethod`, a wrapper for SIMVI's intrinsic latent; needs SIMVI's port to current scvi-tools from [quadbio/simvi](https://github.com/quadbio/simvi)

### Fixed

- Drop the `lightning` git redirect: PyPI has restored the package, and the redirect's tag no longer resolves, which broke installing the `gpu` extra with uv
- `ResolVIMethod` and `scVIVAMethod` use ResolVI and scVIVA from scviva-tools, where they now live, instead of scvi-tools' unmaintained copies; `scviva-tools` joins the `gpu` extra
- Require `pandas<3`: scib-metrics' `Benchmarker.get_results`, used by `IntegrationEvaluator` and `scIBAggregator`, and scarches' scPoli fail on pandas 3
- Install `scarches` from master in development and CI, for anndata 0.13 and a scPoli latent-space bug; PyPI installs keep the latest release
- `scIBAggregator` reads scIB metrics again: the W&B API now returns nested summary entries as dotted keys (`scib.BRAS`), so every run was dropped as missing metrics
- `HarmonyMethod` takes `n_jobs` and `use_gpu`, defaulting to `torch.get_num_threads()` and to CUDA availability; harmony-pytorch's own default uses every physical core of the machine, which fails under Slurm (OMP error #34)
- The `cpu` extra requires `setuptools<82`: louvain, via pyliger, imports `pkg_resources`; without it, importing torch after pyliger fails
- The `cpu` extra requires `bokeh<4`: pyliger imports `umap.plot`, which fails on bokeh 4, so LIGER breaks at import

## [v0.1.0]

### Added

- Basic tests for the base class, gpu and cpu-based methods {pr}`2`
- Added tests for evaluation and aggregatio modules {pr}`3`
- Added basic tutorial {pr}`3`
- Set up documentation on readthedocs {pr}`2`
- Set up precommit.ci
- Push the package to pypi
