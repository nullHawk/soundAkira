# Contributing

Thanks for helping! Bug reports, new components, and docs are all welcome.

## Setup

```bash
git clone <repo> && cd soundakira
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
pytest
```

The core package and the test suite need no GPU and no ML libraries. The end-to-end test uses fake components (`tests/fakes.py`) and needs `ffmpeg`.

## Guidelines

- **Keep the core light.** Model libraries are optional extras, imported inside `Component.load()`.
- **Keep logic pure and tested.** Segmentation, clustering, reference selection and filtering are pure functions; add unit tests for any behaviour change.
- **Keep caching correct.** If a change alters what a stage or component produces, bump its `version` so cached results are invalidated.
- **Make things configurable.** New behaviour goes in `config.py` and `resources/default.yaml`; a test keeps the two in sync.
- **Style.** `ruff check` and `ruff format`; type hints throughout; docstrings explain *why*.

## Commits and PRs

- Conventional, single-sentence commit subjects: `feat: ...`, `fix: ...`, `add: ...`, `update: ...`, `docs: ...`, `test: ...`.
- One logical change per PR, with tests. CI must pass.
- For a new model backend, include in the PR: the extra it needs, its license, and a short note on quality and speed.
