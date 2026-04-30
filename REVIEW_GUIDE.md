# scembed Review Guide

This file is the canonical, agent-neutral source of truth for automated PR review in this repo.
It is written for **agents performing PR reviews on GitHub** — use the imperative voice and be concrete.

**Scope: review only.** Your job is to produce review comments and suggestions on the PR. Do **not** push commits, modify files, or apply fixes yourself. Any changes are the author's call. Flag issues, ask questions, and suggest concrete diffs in comments when helpful — but leave the decision and the edits to the user.

Use `AGENTS.md` for invariants, the source-of-truth map, and development commands. Use this guide for review workflow, risk areas, testing checks, documentation-impact checks, and changed-path test lookup.

## Review-First Workflow

1. Read the PR body first when it is present.
2. Check CI status (`gh pr checks <num>`, `gh run view <run-id> --log-failed`) and investigate any test or lint failures before commenting.
3. Identify changed modules and map them to matching tests (see [Changed-Path Test Lookup](#changed-path-test-lookup)).
4. Check whether the change touches a repo invariant from `AGENTS.md`.
5. Prioritize behavioral regressions, optional-dep gating, and W&B/SDK compatibility over style feedback.
6. Verify that docs (human- and agent-facing) did not become stale — see [Documentation Impact](#documentation-impact).

## High-Risk Areas

- `BaseIntegrationMethod` contract:
  changes to constructor kwargs, `validate_adata`, the abstract `integrate()`
  signature, or the auto-derived `embedding_key` propagate to every method
  subclass and to anything that reads `adata.obsm["X_<name>"]`.
- Optional-dependency gating in `methods/cpu_methods.py` and
  `methods/gpu_methods.py`: every optional import must be preceded by a
  matching `check_deps(...)` call, and new keys must be registered in
  `CHECKERS` (`src/scembed/check.py`).
- W&B SDK drift in `aggregation.py`: the helper methods that parse run
  configs and history are the SDK-version isolation layer; changes there can
  silently break sweep result aggregation against older or newer `wandb`.
- scIB benchmark configuration in `evaluation.py`: silent changes to the
  metric set, neighbor graph, or embedding lookup can shift benchmark
  numbers without any test failure.
- Public API surface in `src/scembed/__init__.py`: every new re-export is a
  long-term commitment. Prefer keeping internals private and importing from
  the submodule.
- Method-name → embedding-key mapping: anything that derives `name` from the
  class name affects every saved embedding. Renaming a method class
  silently changes which `obsm` key the result lands in.

## Changed-Path Test Lookup

Test files mostly mirror module layout under `src/scembed/`:

| Changed path | Tests to check |
|---|---|
| `src/scembed/methods/base.py` | `tests/methods/test_base.py` |
| `src/scembed/methods/cpu_methods.py` | `tests/methods/test_cpu_methods.py` |
| `src/scembed/methods/gpu_methods.py` | `tests/methods/test_gpu_methods.py` |
| `src/scembed/evaluation.py` | `tests/test_evaluation.py` |
| `src/scembed/aggregation.py` | `tests/test_aggregation.py` |
| `src/scembed/factory.py` | `tests/test_basic.py` |
| `src/scembed/utils.py` | `tests/test_basic.py`, `tests/test_retrival.py` |
| `src/scembed/check.py` | `tests/test_basic.py` |

Cross-cutting fixture changes: inspect `tests/conftest.py` and any test
that uses the modified fixture.

## Testing

Apply these checks whenever the PR touches code or tests.

**New code.** Confirm that new behavior is covered by tests.
- Reuse fixtures from `tests/conftest.py` rather than creating parallel ones.
- Prefer `pytest.mark.parametrize` over many near-identical tests.
- Favor few meaningful tests over many redundant ones; flag low-value tests
  that only duplicate existing coverage.

**Failing tests.** If CI is red, do not wave it through.
- Inspect which tests fail and why (`gh pr checks`, `gh run view --log-failed`).
- Distinguish critical regressions (method outputs, optional-dep gating,
  W&B parsing) from trivial or flaky failures.
- Surface critical failures back to the author and ask them to fix before
  merge.

**Modified tests.** Scrutinize *how* existing tests were changed.
- PRs that only relax thresholds, remove assertions, delete cases, or loosen
  `parametrize` matrices are a red flag — tests-working-around-tests defeats
  the purpose.
- Require an explicit justification in the PR body for any weakened
  assertion; do not accept silently.

## Documentation Impact

A single behavioral or API change often touches docs in multiple places.
Check both audiences and ask the author to update what is stale. Point to
the **owning file** for each topic rather than duplicating content in your
review.

**Human-facing docs (`docs/`, Sphinx/RTD).**
- API signature or public symbol changes → `docs/api.md` and autosummary
  entries; any prose referencing the symbol.
- Contributor workflow, environment, or CI changes → `docs/contributing.md`.
- New methods, extras, or behavioral changes → `README.md` and
  `docs/notebooks/basic_usage.ipynb`.

**Agent-facing docs (repo root and `.github/`).**
- Invariants or development commands changed → `AGENTS.md` (Critical
  Invariants, Development Commands).
- Review workflow, risk areas, or testing conventions changed →
  `REVIEW_GUIDE.md` (this file).
- Repo structure, new top-level docs, or moved pointers → `AGENTS.md`
  "Where To Find What" table, `CLAUDE.md`, `.github/copilot-instructions.md`.

If behavior changes but the relevant docs do not, call it out explicitly
in the review and request the update.

## Review Checklist

- Does the change preserve the invariants in `AGENTS.md`?
- Does CI pass, and were any failures investigated? (See [Testing](#testing).)
- Is test coverage adequate and non-redundant, and are modified tests not
  simply weakened? (See [Testing](#testing).)
- For new methods: does the class subclass `BaseIntegrationMethod`, gate
  imports through `check_deps`, register a `CHECKERS` entry, and produce
  the expected `adata.obsm["X_<name>"]` key?
- Does it change scIB benchmark configuration, embedding lookup, or W&B
  parsing in a way that needs explicit release-note-style mention in the
  PR?
- Are all affected human- and agent-facing docs updated? (See
  [Documentation Impact](#documentation-impact).)
- Is the PR scope tight — no unrelated changes bundled in — and is the
  public API surface kept minimal?

## PR Metadata

This repo uses a structured PR template.

Reviewers and agents should treat these sections as the preferred summary
surface:
- summary
- behavior or invariants changed
- tests run
- reviewer focus
- context
- open questions or follow-ups
