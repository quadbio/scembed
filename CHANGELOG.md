# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [Unreleased]

### Fixed
- Drop the `lightning` git redirect: PyPI has restored the package, and the redirect's tag no longer resolves, which broke installing the `gpu` extra with uv
- `ResolVIMethod` and `scVIVAMethod` use ResolVI and scVIVA from scviva-tools, where they now live, instead of scvi-tools' unmaintained copies; `scviva-tools` joins the `gpu` extra

## [v0.1.0]

### Added
- Basic tests for the base class, gpu and cpu-based methods {pr}`2`
- Added tests for evaluation and aggregatio modules {pr}`3`
- Added basic tutorial {pr}`3`
- Set up documentation on readthedocs {pr}`2`
- Set up precommit.ci
- Push the package to pypi
