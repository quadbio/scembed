# AGENTS.md — scembed

`scembed` is a Python package for benchmarking single-cell data integration
methods with [scIB metrics][]. It wraps common integration tools (Harmony,
LIGER, Scanorama, scVI, scANVI, scPoli, ResolVI, scVIVA), drives benchmarks
through [scIB metrics][], and aggregates W&B sweep results produced via
[slurm_sweep][].

[scIB metrics]: https://scib-metrics.readthedocs.io/
[slurm_sweep]: https://github.com/quadbio/slurm_sweep

## Trust Order

When sources disagree:
1. PR description and changed code
2. This file (`AGENTS.md`)
3. `REVIEW_GUIDE.md`
4. Tests under `tests/`
5. Public docs in `docs/`

This file owns invariants and the where-to-find table. Review-specific
workflow lives in `REVIEW_GUIDE.md`.

## Where To Find What

| Topic | Source of truth |
|-------|-----------------|
| Method registry / dispatch | `src/scembed/factory.py` (`get_method_instance`) |
| Method base class & embedding-key convention | `src/scembed/methods/base.py` (`BaseIntegrationMethod`) |
| CPU-only methods (HVG, LIGER, Scanorama, precomputed) | `src/scembed/methods/cpu_methods.py` |
| GPU/torch methods (Harmony, scVI, scANVI, scPoli, ResolVI, scVIVA) | `src/scembed/methods/gpu_methods.py` |
| scIB benchmarking | `src/scembed/evaluation.py` (`IntegrationEvaluator`) |
| W&B sweep aggregation | `src/scembed/aggregation.py` (`scIBAggregator`) |
| Optional-dependency gating | `src/scembed/check.py` (`check_deps`, `CHECKERS`) |
| Public API surface | `src/scembed/__init__.py` |
| Contributor guide | `docs/contributing.md` |
| Usage examples | `docs/notebooks/basic_usage.ipynb` |
| PR review workflow & risk areas | `REVIEW_GUIDE.md` |
| Test fixtures | `tests/conftest.py` |
| Sweep driver | <https://github.com/quadbio/slurm_sweep> |

## Review Guidelines

For GitHub PR reviews, use `REVIEW_GUIDE.md` as the canonical review workflow
and source of review-specific risk areas, testing checks, and
documentation-impact checks. This file only owns the project invariants and
the source-of-truth map above.

## Critical Invariants

- Every integration method subclasses `BaseIntegrationMethod` and writes its
  result into `adata.obsm["X_<name>"]`, where
  `name = ClassName.replace("Method", "").lower()`. Examples:
  `HarmonyMethod` → `X_harmony`, `scVIMethod` → `X_scvi`.
- New methods register their dependency keys in `CHECKERS`
  (`src/scembed/check.py`) and call `check_deps(...)` **before** importing
  the optional package. This is what gives users a clear error when an extra
  is missing.
- Optional install extras and the methods they enable:
  - **`[cpu]`** — `pyliger`, `scanorama`. Used by `LIGERMethod`,
    `ScanoramaMethod`. (`HVGMethod` and `PrecomputedEmbeddingMethod` need
    no extra.)
  - **`[gpu]`** — `harmony-pytorch`, `scvi-tools`, `scarches`, `torch`. Used
    by `HarmonyMethod`, `scVIMethod`, `scANVIMethod`, `scVIVAMethod`,
    `ResolVIMethod`, `scPoliMethod`. **`scPoliMethod` depends on `scarches`,
    not `scvi-tools`.**
  - **`[fast-metrics]`** — `faiss-cpu`, `rapids-singlecell`.
- W&B SDK version differences are isolated in helper methods inside
  `aggregation.py`. Don't sprinkle SDK-version checks elsewhere.
- Public API is whatever `src/scembed/__init__.py` exports
  (`IntegrationEvaluator`, `scIBAggregator`, `get_method_instance`, `logger`,
  `methods`). Keep that surface minimal — submodule symbols stay private
  unless explicitly re-exported.
- Tests mirror the module layout: `src/scembed/X/` → `tests/X/`. Method
  tests live in `tests/methods/`; everything else is flat under `tests/`.

## Development Commands

Python 3.11 and 3.14 are the matrix endpoints (see
`[tool.hatch.envs.hatch-test.matrix]`).

```bash
uv sync                             # install with default groups
uvx hatch test                      # run tests on highest matrix Python
uvx hatch test --all                # full matrix (3.11, 3.14)
uvx hatch run docs:build            # build Sphinx docs
uvx pre-commit run --all-files      # lint + format (biome, pyproject-fmt, ruff)
```
