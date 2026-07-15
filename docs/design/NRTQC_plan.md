# NRT QC Module — Implementation Plan

Step-by-step plan for implementing the NRT QC module specified in
`NRTQC_spec.md`. Each phase is a separate `feature/*` branch off `develop`,
fully tested (unit tests + ruff) and merged before the next phase starts, so
the library stays releasable throughout.

## Phase overview

| Phase | Branch | Deliverable | Depends on |
|-------|--------|-------------|------------|
| 1 | `feature/nrtqc-config` | Config class, schema, templates | — |
| 2 | `feature/nrtqc-utils` | EOS-80 σ₀ + flag utilities | — |
| 3 | `feature/nrtqc-items-basic` | QC items: date, location, ranges, stuck value | 1 |
| 4 | `feature/nrtqc-items-vertical` | QC items: pressure, spike, gradient, rollover | 1 |
| 5 | `feature/nrtqc-items-density` | QC items: density inversion, temp→psal | 1, 2 |
| 6 | `feature/nrtqc-module` | nrtqc package steps 1–3 (end-to-end output) | 3–5 |
| 7 | `feature/nrtqc-compare` | step4 flag comparison report | 6 |
| 8 | `feature/nrtqc-interface` | Public interface + docs | 6, 7 |
| — | `release/v0.5.0` | CHANGELOG, version bump, release | all |

Phases 3–5 are independent of each other and can be done in any order.

## Phase 1 — Configuration foundations

Everything later is config-driven, so the config layer comes first.

- `src/aiqclib/common/config/nrtqc_config.py`: `NRTQCConfig(ConfigBase)`
  resolving `qc_variable_set`, `qc_item_set`, `step_class_set`,
  `step_param_set`, `path_info` for a selected `data_sets` entry
  (mirror `classify_config.py`).
- `src/aiqclib/common/config/yaml_schema.py`: `get_nrtqc_config_schema()` —
  `variables` require only `name` (`flag`, `pos_flag_values`,
  `neg_flag_values` optional); `items` require `name`, `params` free-form
  per item.
- `src/aiqclib/common/config/yaml_templates.py`: `get_config_nrtqc_template()`
  with the §7 sketch from the spec (Mediterranean example).
- Config helpers on `NRTQCConfig`: `get_qc_items()` (enabled items with
  merged default/override params), `get_variable_flag(name)` (existing-flag
  column or None, reusing the `is_flag_missing` convention).
- Registry/loader stubs: `nrtqc_registry.py` / `nrtqc_loader.py` created in
  Phase 6; not needed here.

**Tests:** `tests/test_common_config_nrtqc.py` — valid/invalid YAML against
the schema, set resolution, param merging, optional-flag handling.
**Done when:** template YAML validates and every helper resolves correctly.

## Phase 2 — Shared utilities

- `src/aiqclib/common/utils/seawater.py`: EOS-80 (UNESCO 1983) potential
  density σ₀ as a vectorised function over polars/numpy columns
  (`temp`, `psal`, `pres`), following Fofonoff & Millard (1983).
- `src/aiqclib/common/utils/qc_flags.py`: flag constants (`GOOD = 1`,
  `PROBABLY_BAD = 3`, `BAD = 4`), severity ordering, and a
  `worst_flag(exprs)` helper for aggregation.

**Tests:** `tests/test_common_utils_seawater.py` — σ₀ against published
UNESCO check values and a tolerance test on typical Arctic/Baltic/Med
profiles; `tests/test_common_utils_qc_flags.py` — severity aggregation.
**Done when:** σ₀ matches reference values to expected precision.

## Phase 3 — QC item feature classes: basic tests

`FeatureBase` subclasses under `src/aiqclib/prepare/features/` (one module
per item), registered in `FEATURE_REGISTRY` with `qc_`-prefixed names.
Shared plumbing established here:

- `src/aiqclib/prepare/features/qc_item_base.py`: thin intermediate base
  `QCItemFeatureBase(FeatureBase)` handling the common pattern — compute
  flag column(s) over `filtered_input` with polars window expressions over
  (`platform_code`, `profile_no`), honour `fail_flag` from `feature_info`,
  and join down to `selected_rows[target_name]` row_ids when running inside
  the prepare pipeline (full-frame output when run by the NRT QC module).
- Items (spec §4.1–4.4, §4.9):
  - `qc_impossible_date.py` (`qc_impossible_date`)
  - `qc_impossible_location.py` (`qc_impossible_location`)
  - `qc_global_range.py` (`qc_global_range`)
  - `qc_regional_range.py` (`qc_regional_range`, same check as global with
    config-supplied bounds)
  - `qc_stuck_value.py` (`qc_stuck_value`)
- Register all in `FEATURE_REGISTRY`; built-in default thresholds per spec.

**Tests:** `tests/test_prepare_qc_basic.py` — synthetic mini-profiles per
item: pass/fail/edge cases (null timestamp/position, boundary values,
single-observation profile exemption for stuck value, `fail_flag` override).
**Done when:** each item emits correct 1/3/4 columns, never null.

## Phase 4 — QC item feature classes: vertical-profile tests

Items needing within-profile ordering and neighbour stencils (spec §4.5–4.8):

- `qc_pressure_increasing.py` — constant-run and reversal detection.
- `qc_spike.py` — V1/V2/V3 stencil, depth-dependent thresholds.
- `qc_gradient.py` — same stencil, gradient formula.
- `qc_digit_rollover.py` — adjacent-difference thresholds.

All vectorised via `shift(±1)` over profile windows sorted by `pres`;
profile-boundary observations untestable → flag 1.

**Tests:** `tests/test_prepare_qc_vertical.py` — crafted profiles: clean
monotonic, constant-pressure run, reversed segment, isolated spike vs steep
gradient (spike passes gradient / fails spike and vice versa), rollover jump,
shallow-vs-deep threshold switch at 500 db, profile boundaries.
**Done when:** stencil results match hand-computed expectations row by row.

## Phase 5 — QC item feature classes: density & propagation

- `qc_density_inversion.py` — σ₀ from Phase 2, both-direction comparison
  with threshold, flags temp and psal jointly (two output columns).
- `qc_temp_to_psal.py` — propagation item; consumes the aggregated temp
  flag, so its full effect is exercised in Phase 6 aggregation (unit-tested
  here against precomputed flag columns).

**Tests:** `tests/test_prepare_qc_density.py` — stable profile (pass),
constructed inversion above/below threshold, joint temp+psal flagging;
propagation of 3 and 4 (not 1) onto psal.
**Done when:** inversion cases match hand-computed σ₀ differences.

## Phase 6 — NRT QC module (steps 1–3, end-to-end output)

New package `src/aiqclib/nrtqc/` plus loader/registry wiring:

- `step1_read_input/` — `InputDataSetAll` subclassing prepare's input base
  (validation + preprocessing reused; no row filtering by default).
- `step2_run_qc/` — `QCDataSetAll`: resolve enabled items from
  `get_qc_items()` via `FEATURE_REGISTRY`, instantiate each with its merged
  params, join produced columns onto the data; write intermediate parquet.
- `step3_concat_flags/` — `ConcatDataSetAll`: per-variable
  `{var}_nrt_flag` = worst applicable item flag, apply `qc_temp_to_psal`
  last, write final output parquet (original columns + item columns +
  final flags).
- `src/aiqclib/common/loader/nrtqc_registry.py` + `nrtqc_loader.py`
  (`load_step1_input_dataset`, `load_step2_qc_dataset`,
  `load_step3_concat_dataset`, …) mirroring the classify pair.

**Tests:** `tests/test_nrtqc_step1_input.py`, `..._step2_qc.py`,
`..._step3_concat.py` + `tests/test_nrtqc_pipeline.py` integration test on
the existing CTD fixture: output row count equals input, all configured
columns present, final flags consistent with item columns.
**Done when:** the full read→qc→concat chain produces the §6 output on the
test fixture.

## Phase 7 — Flag comparison report (step 4)

- `step4_compare_flags/` — `CompareFlagsAll` (spec §6.1): per configured
  variable, contingency table (existing × new), optional binary agreement
  metrics via `pos_flag_values`/`neg_flag_values`, per-item breakdown;
  one TSV per variable. Skip variables without `flag`; raise
  `ColumnNotFoundError` if a configured flag column is absent.

**Tests:** `tests/test_nrtqc_step4_compare.py` — with/without `flag`
configured, metrics only when pos/neg values set, known contingency counts
on a synthetic frame, missing-column error.
**Done when:** report values match hand-computed tables.

## Phase 8 — Public interface & docs

- `src/aiqclib/interface/nrtqc.py`: `run_nrt_qc(config)` orchestrator
  (mirroring `create_training_dataset`), running steps 1–4 (step 4 only
  when any variable has a configured flag).
- Expose config helpers in `interface/config.py` (template + read/validate
  for the NRT QC config) and re-export in `aiqclib.interface.__init__`.
- Docs: `docs/source/configuration/nrtqc.rst` (config reference),
  short how-to `docs/source/how-to/nrt_qc.rst`, toctree entries,
  `sphinx-apidoc` regeneration.
- README: add the module to the module list.

**Tests:** `tests/test_interface_nrtqc.py` — end-to-end run via the public
API on the test fixture, with and without comparison step.
**Done when:** docs build clean and the public API round-trips the template.

## Release — v0.5.0

Per `RELEASING.md`: ruff + apidoc → CHANGELOG (`### Added` one-liner for the
NRT QC module) → bump `pyproject.toml` to 0.5.0 → `uv sync` →
`release/v0.5.0` branch → merge to `main` + tag → back-merge to `develop` →
GitHub release → PyPI.

## Conventions (all phases)

- No stdout/stderr from the library.
- Polars expressions only; no per-profile Python loops.
- Run only the tests touched by the phase, plus the Phase 6+ integration
  tests once they exist.
- `uv run ruff check/format` on changed files before each commit.
