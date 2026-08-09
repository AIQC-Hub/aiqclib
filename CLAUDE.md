# CLAUDE.md

`aiqclib` — config-driven ML library for anomaly detection in CTD ocean data. See `README.md` for the full workflow and public API. Dev setup is in `CONTRIBUTING.md`; the release process in `RELEASING.md`. Source in `src/aiqclib/` (`prepare/`, `train/`, `classify/`, `common/`, `interface/`). Internal design/planning docs (outside the Sphinx build) live in `docs/design/` — see its `README.md`.

## Environment & dependencies

- Use **`uv`** for everything: `uv sync` installs deps, `uv run <cmd>` runs in the env. `uv` handles all dependencies — no separate install step needed.
- `mamba`/conda is only used to install `uv` itself into `base`. It is **optional**; skip it if `uv` is already available.
- Prefix commands with `uv run` (e.g. `uv run pytest`, `uv run ruff check src`).

## Tests

- The suite is large and slow — **do not run the whole thing routinely.** Run only the tests relevant to your change, e.g. `uv run pytest tests/test_prepare_features.py`.
- Tests need fixtures under `tests/data/` (not in git): `bash scripts/fetch_test_data.sh` (once). See `tests/tests_README.md`.
- Test files mirror the pipeline stages (`test_prepare_*`, `test_training_*`, `test_classify_*`, `test_common_*`, `test_interface_*`).

## Lint & format

- `uv run ruff check src` / `uv run ruff format src` (and likewise for `tests`).

## Git (gitflow)

- Follows **gitflow**: `main` (releases), `develop` (integration), plus `feature/*`, `release/*`, `hotfix/*` branches.
- Branch off `develop` for features (`feature/<name>`); branch off `main` for hotfixes.
- Do **not** commit directly to `main` or `develop` — merge via the appropriate feature/release/hotfix branch.

## Releasing

- Full checklist in `RELEASING.md`. Order: (ruff + `sphinx-apidoc` if needed) → update `CHANGELOG.md` → bump `version` in `pyproject.toml` → `uv sync` (updates `uv.lock`) → GitHub release triggers PyPI publish.
- Keep `CHANGELOG.md` **compact**: terse one-line bullets under `### Added/Changed/Fixed`, added to the `## [Unreleased]` section (Keep a Changelog style).
