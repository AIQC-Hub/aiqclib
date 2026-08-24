# Profile-Level Mode: Specification

Specification for the **profile-level** preparation, training, and
classification mode of `aiqclib`. The existing pipeline works at the
**observation level** (one row per CTD observation); this mode produces,
trains on, and classifies **one row per profile** (a CTD cast), so models
can flag whole casts instead of single observations.

## 1. Definitions

- **Profile**: the set of observations sharing the composite key
  `["platform_code", "profile_no"]` (`PROFILE_KEYS`). This is already the
  grouping key used by summary statistics, the QC items, and profile
  selection.
- **Observation**: one row of the input parquet, identified by
  `["platform_code", "profile_no", "observation_no"]` (`OBSERVATION_KEYS`).
- **Level of a feature**: whether a feature produces one value per
  observation (`observation`) or one value per profile (`profile`).
  Declared as a class attribute `level` on every feature class
  (default `"observation"` on `FeatureBase`).

Profile-level feature classes (one value per profile, usable directly):

| Feature | Why profile-level |
| --- | --- |
| `location` | longitude/latitude are profile metadata |
| `day_of_year` | derived from `profile_timestamp` |
| `profile_summary_stats` | per-profile summary statistics by construction |
| `qc_impossible_date` | checks `profile_timestamp` |
| `qc_impossible_location` | checks longitude/latitude |
| `qc_position_on_land` | RTQC4 checks the depth at the profile position |
| `qc_stuck_value` | RTQC13 compares all values of a profile |

All other features (`basic_values`, `flank_up`, `flank_down`, and the
remaining `qc_*` items) are observation-level. Note that
`qc_pressure_increasing` is **observation-level** despite its profile-style
output column name: its fail expression compares each row against its
predecessor / running maximum, so values differ within a profile.

## 2. Class labels

Per target variable, controlled by a new optional `label_mode` key in
`target_sets[].variables[]` (default `binary`):

- **`binary`**: the profile label is `1` if **any** observation of the
  profile has a flag in `pos_flag_values`, else `0` (dtype `UInt32`).
  Trains with the existing classifier models.
- **`proportion`**: the profile label is the **fraction of bad-flagged
  observations** in the profile, a `Float64` in `[0, 1]`:

  ```
  label = n_pos / n_valid
  n_pos   = observations with flag ∈ pos_flag_values
  n_valid = observations with flag ∈ pos_flag_values ∪ neg_flag_values
  ```

  Trains with the new **regressor** models (`XGBoostRegressor`,
  `RandomForestRegressor`).

**Denominator decision**: `n_valid` counts only observations carrying a
valid (positive or negative) flag, consistent with the observation-level
pipeline, which only ever trains on such rows. Profiles with
`n_valid == 0` are dropped (with a logged count). *Rejected alternative*:
dividing by all observations of the profile; this would silently dilute
the proportion with unflagged observations; it remains a one-expression
change in `LocateDataSetProfile` if ever needed.

## 3. Feature rules at profile level

The profile extract step (`ExtractDataSetProfile`) resolves each entry of
`feature_param_sets[].params` against the feature class's `level`:

- **`level == "profile"`**: the feature runs unchanged against the
  one-row-per-profile frame and its columns are used directly.
- **`level == "observation"`**: the entry **must** carry a new `agg` key
  (list of aggregation names). The feature runs against the selected
  observation rows, then each feature column is aggregated per profile,
  producing columns named `{column}_{agg}`. An observation-level feature
  without `agg` is **rejected** with a `ValueError` naming the feature.
  This is the rule that excludes un-aggregated observation-level NRT QC
  items (e.g. `qc_spike`) from profile-level training.

Available aggregations (registry in `common/utils/aggregations.py`):

| Name | Meaning |
| --- | --- |
| `mean`, `min`, `max`, `median`, `std`, `sum`, `first` | plain polars aggregations |
| `fail_frac` | fraction of rows where the (QC flag) column ≠ `FLAG_GOOD` |
| `fail_any` | 1 if any row has the column ≠ `FLAG_GOOD`, else 0 (Int8) |

Normalization keeps the existing fit/apply flow: features are scaled per
observation first (unchanged `scale_first`/`scale_second`), then
aggregated, so `{col}_{agg}` columns of scaled values stay in range but
are not themselves re-fitted.

## 4. Configuration

New/changed config surface (all optional; existing configs are unaffected):

- `target_sets[].variables[].label_mode: binary | proportion` (dataset,
  classification, and training schemas; default `binary`).
- `feature_param_sets[].params[].agg: [name, ...]` (dataset and
  classification schemas).
- New step classes selected via the usual `step_class_sets`:

  ```yaml
  step_class_sets:
    - name: profile_steps
      steps:
        input: InputDataSetA          # reused
        summary: SummaryDataSetA      # reused
        select: SelectDataSetAll      # reused
        locate: LocateDataSetProfile  # new
        extract: ExtractDataSetProfile  # new
        split: SplitDataSetProfile    # new
  ```

- New models: `XGBoostRegressor` (`XGBR`), `RandomForestRegressor`
  (`RFR`) for proportion labels. A `ModelSuite` mixing regressors and
  classifiers is rejected at load time.
- New config templates / stages: `prepare_profile`, `train_profile`,
  `classify_profile`.

Schema fix folded in: `feature_param_sets[].params[]` now allows the
documented QC-item keys `params` and `fail_flag`, and `col_names` is no
longer required (QC items derive variables from `params`).

## 5. Pipeline outputs per step (profile mode)

### Preparation

| Step | Class | Output |
| --- | --- | --- |
| 1-3 | existing classes | unchanged (`selected_profiles.parquet`) |
| 4 | `LocateDataSetProfile` | `selected_rows_{target}.parquet`, one row per profile: `row_id, profile_id, platform_code, profile_no, label, pair_id` (no `observation_no`; `pair_id` empty). Also `selected_observation_rows_{target}.parquet`, the valid-flagged observation rows (LocateDataSetAll schema) kept for feature aggregation. |
| 5 | `ExtractDataSetProfile` | `extracted_features_{target}.parquet`: profile spine + direct profile features + `{col}_{agg}` aggregated features. |
| 6 | `SplitDataSetProfile` | `train_set_{target}.parquet` / `test_set_{target}.parquet`: stratified split for binary labels (inherited), plain random split + uniform k-fold for proportion labels. |

### Training

No new step classes. The train/classify step classes use shared,
level-tolerant id-column constants (`common/constants.py`), so frames
without `observation_no` pass through unchanged. Binary profile labels
use the existing classifiers; proportion labels use the regressors, whose
`predict()` clips predictions to `[0, 1]` as the `score` and keeps
`predicted_label = (score >= predicted_label_threshold)`; the threshold
now reads "flag the profile if the predicted bad fraction ≥ t". Regressor
reports carry MAE / RMSE / R² / n_samples instead of the classification
report; metric plots show predicted-vs-actual and residuals instead of
ROC/PR.

### Classification

| Step | Class | Output |
| --- | --- | --- |
| 1-3, 6 | existing classes | unchanged |
| 4 | `LocateDataSetProfile` (classify) | one row per **every** profile; label as in preparation, or null when the target has no flag (`skip_evaluation`). |
| 5 | `ExtractDataSetProfile` (classify) | as prepare step 5 with `normalization_role = "apply"`. |
| 7 | `ConcatDataSetProfile` | `predictions_profile.parquet`: one row per profile joined on `PROFILE_KEYS`, columns `{target}_label/_predicted/_score`. Optional step param `broadcast_to_observations: true` joins the predictions back onto the observation-level input instead. |

## 6. Known limitations

- No positive/negative profile **pairing** at profile level (`pair_id` is
  kept empty for schema compatibility); a paired/down-sampled variant is
  deferred.
- The model step class is global per config, so mixing a classifier
  target and a regressor target in one run is not possible; use separate
  configs.
- Profile-level datasets are much smaller than observation-level ones;
  smaller `k_fold` values are recommended.
- Only two regressor wrappers are provided initially; mirroring all nine
  classifier methods is deferred.
