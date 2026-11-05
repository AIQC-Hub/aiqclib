# Moving NRT QC to `nrtqclib`: Migration Plan

Plan for moving the NRT QC module out of `aiqclib` into a new package,
`nrtqclib` (repo `AIQC-Hub/nrtqclib`). Drafted 2026-11-05 against `aiqclib`
0.13.0 (`develop` @ `adebc5c`).

## Decision

`nrtqclib` **depends on `aiqclib`**. It owns the NRT QC pipeline, its config,
docs and tests. The QC items themselves (`prepare/features/qc_*.py`) and the
shared framework stay in `aiqclib`, so the tests that produce NRT flags and
the tests used as training features are one implementation.

```
nrtqclib  (pipeline, config, batch phase, docs)
   │ depends on
   ▼
aiqclib   (QC items, feature registry, config/dataset base classes, ML)
```

### Why this direction

- One source of truth for the QC items. Training features and operational
  NRT flags cannot drift apart.
- Small move: only the pipeline layer (about 1,290 lines) relocates. The 13
  QC item modules (about 1,070 lines) keep their `FeatureBase` parent.
- Reversible: if a lightweight install becomes important later, the QC items
  can be moved down into `nrtqclib` (and `aiqclib` made to depend on it)
  without changing `nrtqclib`'s public API.

### Accepted cost

Installing `nrtqclib` installs the full `aiqclib` stack (xgboost, shap,
numba, scikit-learn, matplotlib). Phase 1 makes `aiqclib` imports lazy so
that at least *loading* `nrtqclib` does not import them; today
`import aiqclib.prepare.features.qc_spike` takes about 3.6 s and pulls in
xgboost, sklearn and matplotlib through `aiqclib/__init__.py`.

## What moves and what stays

### Moves to `nrtqclib`

| `aiqclib` today | `nrtqclib` |
| --- | --- |
| `nrtqc/step1_read_input/` | `nrtqclib/step1_read_input/` |
| `nrtqc/step2_run_qc/` | `nrtqclib/step2_run_qc/` |
| `nrtqc/step3_concat_flags/` | `nrtqclib/step3_concat_flags/` |
| `nrtqc/step4_compare_flags/` | `nrtqclib/step4_compare_flags/` |
| `interface/nrtqc.py` (`run_nrt_qc`) | `nrtqclib/interface/nrtqc.py` |
| `common/config/nrtqc_config.py` | `nrtqclib/config/nrtqc_config.py` |
| `get_nrtqc_config_schema()` in `common/config/yaml_schema.py` | `nrtqclib/config/yaml_schema.py` |
| `_get_nrtqc_*()`, `get_config_nrtqc_template()` in `common/config/yaml_templates.py` | `nrtqclib/config/yaml_templates.py` |
| `common/loader/nrtqc_loader.py`, `nrtqc_registry.py` | `nrtqclib/loader/` |
| `"nrt_qc"` stage in `interface/config.py` | `nrtqclib.read_config`, `read_config_template`, `write_config_template` |
| `nrt_qc` phase in `interface/batch.py` (`PHASES`) | `nrtqclib.run_batch` (see Phase 1, step 2) |
| `tests/test_nrtqc_*.py` (5 files), `test_interface_nrtqc.py`, `test_common_config_nrtqc.py` | `tests/` |
| NRT QC cases in `test_common_base_config.py`, `test_interface_config.py`, `test_interface_batch.py`, `test_common_utils_progress.py` | `tests/` |
| `nrtqc_config_001` fixture in `tests/conftest.py` | `tests/conftest.py` |
| `docs/source/how-to/nrt_qc.rst`, `configuration/nrtqc.rst` | `docs/source/` |
| `nrt_qc` mode section of `docs/source/how-to/batch_processing.rst` | `docs/source/how-to/` |
| `docs/source/api/aiqclib.nrtqc*.rst` | regenerated with `sphinx-apidoc` |
| `docs/design/NRTQC_doc.md`, `NRTQC_spec.md`, `NRTQC_plan.md` | `docs/design/` |

### Stays in `aiqclib`

- All QC items: `prepare/features/qc_*.py` (13 modules, including the RTQC4
  `qc_position_on_land` added after 0.11.0) and `qc_item_base.py`.
- `common/loader/feature_registry.py` (the `qc_*` registrations).
- `common/utils/qc_flags.py`, `common/constants.py`
  (`OBSERVATION_KEYS`, `PROFILE_KEYS`).
- The batch engine (`interface/batch.py`, `Phase`) for prepare, train and
  classify.
- `docs/source/how-to/qc_items_as_features.rst` (links to the `nrtqclib`
  docs once they exist).

### `aiqclib` API that `nrtqclib` will rely on

These become a supported surface in Phase 1: no silent renames or signature
changes without a CHANGELOG entry.

| Module | Names |
| --- | --- |
| `aiqclib.common.base.config_base` | `ConfigBase` |
| `aiqclib.common.base.dataset_base` | `DataSetBase` |
| `aiqclib.prepare.step1_read_input.input_base` | `InputDataSetBase` |
| `aiqclib.common.loader.feature_registry` | `FEATURE_REGISTRY` |
| `aiqclib.prepare.features.qc_item_base` | `QCItemFeatureBase`, `OBSERVATION_KEYS` |
| `aiqclib.common.utils.qc_flags` | `FLAG_GOOD`, `FLAG_BAD`, `worst_flag`, `flag_as_int`, `normalize_flag_values` |
| `aiqclib.common.utils.config` | `get_config_item`, `read_config` |
| `aiqclib.common.utils.progress` | `report_progress` |
| `aiqclib.interface.batch` | `Phase` and the batch runner hook from Phase 1 |

The exact list is fixed in Phase 1 by grepping the moved code for
`from aiqclib` imports.

## Phases

Each `aiqclib` phase is a `feature/*` branch off `develop` (gitflow), with
ruff and the relevant tests passing before merge. Release steps follow
`RELEASING.md`.

| Phase | Repo | Branch | Deliverable | Release |
| --- | --- | --- | --- | --- |
| 1 | `aiqclib` | `feature/nrtqc-extension-points` | Pluggable config sections and batch phases, lazy imports, API contract test, deprecation warning | `aiqclib` 0.14.0 |
| 2 | `nrtqclib` | initial import, then `develop` | Repo skeleton, packaging, CI, docs build | none |
| 3 | `nrtqclib` | `feature/move-nrtqc` | Code, tests, docs moved; parity with `aiqclib` 0.14.0 verified | `nrtqclib` 0.1.0 |
| 4 | `aiqclib` | `feature/remove-nrtqc` | NRT QC pipeline removed, docs point to `nrtqclib` | `aiqclib` 0.15.0 |
| 5 | both | n/a | Cross-repo CI and release routine | ongoing |

### Phase 1: prepare `aiqclib` (no behaviour change)

1. **Pluggable config sections.** `ConfigBase.__init__`
   (`common/base/config_base.py:88`) hardcodes a `yaml_schemas` dict that
   includes `"nrt_qc_sets"`, and template lookup goes through
   `get_template_text()` and the `_TEMPLATES` registry in
   `yaml_templates.py`. Turn both into overridable hooks, for example class
   methods `ConfigBase.get_yaml_schema(section_name)` and
   `ConfigBase.get_template_text(name)` that subclasses extend. `nrtqclib`'s
   `NRTQCConfig` then supplies its own schema and template without `aiqclib`
   knowing the section exists. `aiqclib`'s own configs keep the current
   behaviour.
2. **Pluggable batch phases.** `interface/batch.py` defines `PHASES` as a
   fixed tuple that includes the `nrt_qc` phase (`in_all=False`). Let the
   batch runner accept extra phases (for example a `phases=` argument on an
   internal `_run_batch`, with `run_batch` passing the built-in ones), so
   `nrtqclib.run_batch` can call the same engine with its own `nrt_qc`
   `Phase`. Prefer this explicit call over registering a phase as an import
   side effect.
3. **Lazy top-level imports.** Replace the eager imports in
   `src/aiqclib/__init__.py` with a module-level `__getattr__` (PEP 562), so
   `import aiqclib.prepare.features.qc_spike` no longer loads xgboost,
   sklearn or matplotlib. Check that `from aiqclib import train_and_evaluate`
   and the Sphinx autodoc build still work.
4. **API contract test.** Add `tests/test_common_public_api.py` that imports
   every name in the table above. A refactor that would break `nrtqclib`
   then fails in `aiqclib` CI first.
5. **Deprecation warning.** `aiqclib.run_nrt_qc`, `read_config(...,
   stage="nrt_qc")` and `run_batch(mode="nrt_qc")` emit a
   `DeprecationWarning` saying the module is moving to `nrtqclib` and will be
   removed in 0.15.0.
6. CHANGELOG, release `aiqclib` 0.14.0.

### Phase 2: bootstrap `nrtqclib`

A first copy was made on 2026-11-05 (`aiqclib` `develop` @ `adebc5c`): code,
tests and docs copied unchanged into the target layout, plus packaging, CI
and a `CLAUDE.md`. Imports still point at `aiqclib.*` until Phase 3.

1. Mirror `aiqclib`'s setup: `uv`, hatchling, `requires-python = ">=3.12"`,
   MIT license, ruff at the same minor version as `aiqclib`'s lockfile,
   `src/nrtqclib/` layout, `py.typed`.
2. Dependencies: `aiqclib` plus anything the pipeline imports directly
   (polars, pyyaml, jsonschema). During development `aiqclib` comes from the
   sibling checkout (`[tool.uv.sources]`, editable); at release it becomes
   `aiqclib>=0.14,<0.15`.
3. CI: `check_package.yml`, `code_coverage.yml`, `publish_to_pypi.yml`
   adapted from `aiqclib` (no test-data download). Set up the PyPI trusted
   publisher for the new project.
4. Sphinx + Read the Docs project, with intersphinx to the `aiqclib` docs so
   QC item references link across.
5. `README.md`, `CHANGELOG.md`, `CONTRIBUTING.md`, `RELEASING.md`,
   `CLAUDE.md` (same gitflow, compact CHANGELOG and writing-style rules),
   `main` and `develop` branches.

### Phase 3: move the code

1. Rewrite imports in the copied files (`aiqclib.nrtqc.*` to `nrtqclib.*`
   and the other rows of the "Moves" table; shared imports stay
   `aiqclib.*`). Note the source commit in the first `nrtqclib` commit
   message; the history stays in `aiqclib`.
2. `NRTQCConfig` subclasses `aiqclib`'s `ConfigBase` and uses the Phase 1
   hooks for its schema and the `template:nrt_qc_sets` template.
3. Public API in `nrtqclib/__init__.py`: `run_nrt_qc`, `run_batch`,
   `read_config`, `read_config_template`, `write_config_template`.
4. Test data: the NRT QC tests only need
   `tests/data/input/nrt_cora_bo_test.parquet` (64 KB) and
   `tests/data/config/test_dataset_invalid.yaml`. Commit them instead of
   adding a release-hosted fixture archive and fetch script.
5. Move the docs and design docs; regenerate the API pages with
   `sphinx-apidoc`. Rename `aiqclib.nrtqc` references in prose and examples.
6. **Parity check (acceptance criterion).** Run `aiqclib.run_nrt_qc`
   (0.14.0) and `nrtqclib.run_nrt_qc` on the same fixture and config, and
   assert the step 2, 3 and 4 outputs are equal
   (`polars.testing.assert_frame_equal`). Keep this as a one-off script, not a
   permanent test, since `aiqclib` drops the pipeline in Phase 4.
7. Release `nrtqclib` 0.1.0.

### Phase 4: remove NRT QC from `aiqclib`

1. Delete the files in the "Moves" table and the NRT QC entries in
   `yaml_schema.py`, `yaml_templates.py` (`"template:nrt_qc_sets"`),
   `config_base.py`, `interface/config.py`, `interface/batch.py` (the
   `nrt_qc` `Phase`) and `__init__.py`.
2. Remove the NRT QC tests and the `nrtqc_config_001` fixture. Keep the QC
   item feature tests.
3. Docs: drop `how-to/nrt_qc` and `configuration/nrtqc` from `index.rst`;
   rewrite the intro sentence in `index.rst`, the README section and the
   `nrt_qc` part of `how-to/batch_processing.rst` to point to `nrtqclib`;
   update `qc_items_as_features.rst` links; regenerate the API pages.
4. `docs/design/README.md`: replace the NRT QC section with a pointer to the
   `nrtqclib` repo.
5. CHANGELOG under `### Removed`, release `aiqclib` 0.15.0 (breaking, minor
   bump while pre-1.0).
6. In `nrtqclib`, widen the pin to `aiqclib>=0.14,<0.16` once its CI passes
   against 0.15.0, and release 0.1.1.

### Phase 5: keeping the two in step

- **Release order.** A fix to a QC item ships in `aiqclib` first; `nrtqclib`
  then raises its lower bound if it needs the fix. Pure pipeline or config
  changes ship in `nrtqclib` alone. Threshold tuning (Baltic, Arctic) is a
  user config change and needs neither.
- **Upper bound.** Each new `aiqclib` minor needs a `nrtqclib` release that
  widens the pin. To catch breakage early, add a scheduled `nrtqclib` CI job
  that runs the tests against `aiqclib`'s `develop` branch.
- **Contract test.** When `nrtqclib` starts importing a new `aiqclib` name,
  add it to `test_common_public_api.py` in `aiqclib`.
- **Ruff.** Both repos pin the same ruff minor; upgrade them together.

## Open questions

- Keep the Phase 1 deprecation shim, or remove `run_nrt_qc` from `aiqclib`
  without a transition release? (The plan assumes the shim.)
- conda-forge: `nrtqclib` would need its own recipe, and `aiqclib` is not on
  conda-forge yet. Out of scope here.
