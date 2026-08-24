# Profile-Level Mode — Implementation Plan

Step-by-step plan for implementing the profile-level mode specified in
`PROFILE_spec.md`. Each phase is a separate `feature/*` branch off
`develop`, fully tested (unit tests + ruff) and merged before the next
phase starts, so the library stays releasable throughout.

## Phase overview

| Phase | Branch | Deliverable | Depends on |
|-------|--------|-------------|------------|
| 1 | `feature/profile-constants` | Shared id/key constants, feature `level` metadata, level-tolerant drops, schema keys | — |
| 2 | `feature/profile-prepare` | Prepare steps 4–6 profile classes, aggregation registry, template | 1 |
| 3 | `feature/profile-regressors` | Regressor model base + XGBR/RFR wrappers, regression reports/plots | 1 |
| 4 | `feature/profile-classify` | Classify steps 4/5/7 profile classes, template | 2, 3 |
| 5 | `feature/profile-docs` | Sphinx docs, CHANGELOG | 2–4 |

## Phase 1 — Shared constants and level metadata (pure refactor)

- New `common/constants.py`: `PROFILE_KEYS`, `OBSERVATION_KEYS` (moved
  from `prepare/features/qc_item_base.py`, re-exported there for
  backward compatibility), `ID_COLUMNS`, `LABELED_ID_COLUMNS`,
  `existing_columns(df, cols)`.
- Replace the six duplicated `drop_cols`/`test_cols` lists in
  `train/step4_build_model/build_model{,_suite}.py`,
  `train/step2_validate_model/kfold_validation{,_suite}.py`,
  `classify/step6_classify_dataset/dataset_all{,_suite}.py` with the
  shared constants; `.drop(..., strict=False)` and
  `existing_columns(...)` selects make them tolerate frames without
  `observation_no` (also `SplitDataSetAll.add_k_fold` column fronting).
- `FeatureBase.level = "observation"`; override `"profile"` in
  `location`, `day_of_year`, `profile_summary`, `qc_impossible_date`,
  `qc_impossible_location`, `qc_stuck_value`.
- `ModelBase.is_regressor = False` (groundwork for Phase 3).
- Schema: `label_mode` on target variables (all three schemas), `agg` on
  feature params (dataset + classification), and the pre-existing QC-item
  schema fix (`params`, `fail_flag`, `required: [feature]`).
- `ConfigBase.get_label_mode(target_name)` (default `binary`).
- Tests: `test_feature_levels.py` pins every registry entry's level; the
  full suite must stay green (zero behavior change).

## Phase 2 — Prepare pipeline profile mode

- `prepare/step4_select_rows/dataset_profile.py::LocateDataSetProfile`:
  valid-flagged `observation_rows` per target + one-row-per-profile
  `selected_rows` with binary/proportion label per `label_mode`.
- `ExtractFeatureBase` gains an optional `observation_rows` kwarg,
  forwarded by `dataset_loader` and `interface/prepare.py`.
- `prepare/step5_extract_features/dataset_profile.py::ExtractDataSetProfile`:
  level-based branching, `agg` aggregation via `common/utils/aggregations.py`,
  rejection of un-aggregated observation-level features.
- `prepare/features/qc_item_base.py`: profile-keys join branch when
  `selected_rows` has no `observation_no`.
- `prepare/step6_split_dataset/dataset_profile.py::SplitDataSetProfile`:
  random split + uniform folds for proportion labels, inherited
  stratified logic for binary.
- Template `template:data_sets_profile`, stage `prepare_profile`.
- Tests: step 4/5/6 unit tests, `test_dataset_profile_001/002.yaml`
  fixtures, end-to-end prepare→train (binary, XGB).

## Phase 3 — Regressor models

- `common/base/scikit_learn_regressor_base.py::SklearnRegressorModelBase`
  (`is_regressor = True`): clipped `predict`, regression report
  (MAE/RMSE/R²/n_samples), SHAP without class axis,
  `warn_constant_labels` diagnostic.
- Wrappers `train/models/xgboost_regressor.py` (`XGBR`) and
  `train/models/random_forest_regressor.py` (`RFR`), registered in the
  model registries; `ModelSuite` rejects mixed regressor/classifier
  method sets.
- `common/utils/metric_plots.py`: predicted-vs-actual + residual panels
  when `is_regressor`.
- Template `template:training_sets_profile`, stage `train_profile`.
- Tests: `REGRESSOR_CASES` in `_model_cases.py`,
  `test_training_models_regressor.py`, end-to-end proportion
  prepare→train.

## Phase 4 — Classify pipeline profile mode

- `classify/step4_select_rows/dataset_profile.py::LocateDataSetProfile`:
  every profile, label or null (`skip_evaluation`).
- `classify/step5_extract_features/dataset_profile.py::ExtractDataSetProfile`:
  prepare subclass with `normalization_role = "apply"`;
  `observation_rows` threaded through `classify_loader` +
  `interface/classify.py`.
- `classify/step7_concat_datasets/dataset_profile.py::ConcatDataSetProfile`:
  one row per profile joined on `PROFILE_KEYS`,
  `predictions_profile.parquet`, optional `broadcast_to_observations`.
- Template `template:classification_sets_profile`, stage
  `classify_profile`; `interface/batch.py` verified unchanged.
- Tests: step 4/5/7 unit tests, apply-mode normalization test,
  end-to-end classify covering classifier/binary, regressor/proportion,
  and a label-free target; suite variant via config mutation.

## Phase 5 — Docs and changelog

- New `docs/source/how-to/profile_level_pipeline.rst`; updates to
  configuration pages, `qc_items_as_features.rst`,
  `prediction_threshold.rst`, `algorithm_selection.rst`; sphinx-apidoc
  regeneration.
- `CHANGELOG.md` bullets under `[Unreleased]`.
