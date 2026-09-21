# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
As this project is still in active development, it does not yet strictly adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- `derived_values` feature: sigma-0, depth and potential temperature computed from the measured variables, or passed through when the input already carries them.
- `stratification` feature: the vertical density gradient, N squared, its magnitude and an unstable-layer flag.
- `profile_smooth` feature: Savitzky-Golay smoothed value, first and second derivative, residual, robust z-score of the residual, curvature-to-residual ratio, Argo spike index, and the window fractions of outliers and high curvature.
- `neighbor_diff` feature: differences against the levels above and below at configurable lags, plus the window fraction of large differences.
- `regime_flags` feature: in-mixed-layer, in-gradient-layer (thermocline or halocline) and normalized depth to the peak gradient.
- `geo_context` feature: bathymetry and coast distance read as input columns (no GEBCO or GSHHG reader), plus normalized depth, distance to bottom and a deep stable layer flag.
- `rolling_stats` feature: mean, median, MAD, min, max, standard deviation and the local robust z-score over configurable centred windows.
- `profile_summary_stats` accepts `mad`, computed per profile from the input since the step 2 summary table does not carry it.
- `common.utils.profile_signal`: Savitzky-Golay smoothing and derivatives, neighbour differences, central gradients, the shared RTQC9 spike stencil, and centred rolling, MAD and robust z statistics, all partitioned per profile.
- `common.utils.seawater`: `gravity`, `depth_from_pressure` and `brunt_vaisala_squared`.
- Feature entries accept an `outputs` list naming which columns of a feature group to emit.
- Documentation: a page per new feature group under Features, and a how-to, `Building a Feature Set from the Profile Shape`, on choosing among them.

## [0.12.2] - 2026-09-21
### Fixed
- The density inversion test (RTQC14) no longer computes a density from values that cannot be measurements. A netCDF fill value or a placeholder such as -999 used to run through the equation of state, raising numpy overflow and invalid-value `RuntimeWarning`s during a run, and, where it did not overflow, producing a finite density far outside sea water that made the good observation next to it read as an inversion. Such inputs are now treated as missing, which the test already counts as a pass; the accepted ranges are much wider than the global range test, so no real measurement changes

## [0.12.1] - 2026-09-21
### Added
- NRT QC comparison report gains an `item_breakdown_contingency` section: per item, the cross-tabulation of existing flag value against the item's own flag value, so an item using more than one failing value is not collapsed into a single count as in `item_breakdown`

## [0.12.0] - 2026-08-25
### Added
- Profile-level pipeline: `LocateDataSetProfile` / `ExtractDataSetProfile` / `SplitDataSetProfile` (prepare) and `LocateDataSetProfile` / `ExtractDataSetProfile` / `ConcatDataSetProfile` (classify) produce, train on, and classify one row per profile instead of one per observation
- Per-target `label_mode`: `binary` (any bad observation, default) or `proportion` (fraction of bad-flagged observations, in [0, 1]) for profile-level labels
- Regressor models `XGBoostRegressor` (`XGBR`) and `RandomForestRegressor` (`RFR`) for proportion labels, with regression reports (MAE/RMSE/R²) and predicted-vs-actual metric plots; `ModelSuite` rejects mixed classifier/regressor method sets
- Feature classes declare a `level` (`observation`/`profile`); observation-level features aggregate per profile via a new `agg` feature-param key (`mean`, `min`, `max`, `median`, `std`, `sum`, `first`, `fail_frac`, `fail_any`), and un-aggregated observation-level features are rejected at profile level
- Configuration templates and stages `prepare_profile`, `train_profile`, `classify_profile`; how-to guide `profile_level_pipeline`
- Shared identity-column constants (`aiqclib.common.constants`) replace six duplicated `drop_cols`/`test_cols` lists in the train/classify steps
- New NRT QC item `position_on_land` (RTQC4), flagging profiles whose position is not in the ocean. It reads an externally computed sea floor depth column already in the input (`depth_column`, default `bathymetry`, deliberately not `depth`, which is the measurement depth) rather than an external bathymetry grid, with `positive_depth` (default `true`) saying which sign means deeper. Absent from the config templates, and a missing column raises rather than passing every row
- `run_batch` accepts `mode="nrt_qc"`, running the NRT QC module over a table of datasets from an `nrt_qc_set_name` column and an `nrt_qc_config` file. It is not part of `mode="all"`, which still runs prepare, train and classify: NRT QC flags are an input to the prepare phase rather than a step of it
- NRT QC items accept `include_in_final_flag` (default `true`): an item set to `false` still runs and still writes its flag column, but no longer feeds the aggregated `{variable}_nrt_flag`, so a test can be recorded without deciding the verdict

### Changed
- The SHAP cost heads-up is now a `[aiqclib] note:` line instead of a `UserWarning`, so it no longer reads as a fault in the library. Same message, still once per run and still printed whether or not `verbose` is set
- Saved models record the XGBoost version that wrote them, and loading one compares it. XGBoost's own warning fires on any version difference, down to the patch release and in both directions, from inside `pickle` and naming no file; only one direction matters, so loading into the same version or newer is now a `[aiqclib] note:` needing no action, and loading into an older one is a warning naming both versions. Model files written before this carry no version and keep a warning saying the direction cannot be checked
- The GPU how-to documents the XGBoost version mismatch, with measured predictions for one model loaded under six releases: loading into an older XGBoost than trained the model moved scores by up to 0.076 and flipped 21 of 2,000 labels, while loading into a newer one reproduced them bitwise
- Em and en dashes removed throughout the docs, docstrings and comments, replaced with ordinary punctuation; the rule is recorded in `CLAUDE.md`
- Sphinx `smartquotes_action` set to `"qe"`, so `--` and `---` are no longer rendered as dashes in the HTML

### Fixed
- The configuration schema rejected the documented QC-item feature keys `params` and `fail_flag`, and required `col_names` even though QC items derive their variables from `params`

## [0.11.0] - 2026-08-15
### Added
- Computing SHAP values over more than 100,000 rows now warns once per run, naming `calculate_shap`; it is normally the largest cost in a run (~half of a training phase, ~99% of a classification phase) and nothing in the output attributed the time to it
- The SHAP how-to has a "What It Costs" section with measured shares, how the cost scales, and why turning it off is a bigger lever than a GPU
- `model_params` may mix shared parameters with per-model sections: a key naming a model (long or short form) applies only to that model, plain keys apply to all, and a model's own section overrides the shared value
- New how-to page on GPU acceleration: which parts of the pipeline can use one (everything XGBoost does: fitting, prediction and SHAP), the `device: cuda` setting for single models and for `ModelSuite`, why saved models stay usable on CPU-only machines, what to check before running in a container, and why an older GPU may need an `xgboost` version pin, since a wheel carries code only for the GPU generations it was built for, and a too-new one fails at fit time with `This program was not compiled for SM 60`. Includes a measured comparison: 1.93x on a train phase, with the whole saving coming from SHAP rather than fitting, and why `GPUTreeExplainer` is not used: 28x on `RandomForest`, but slower than the ordinary explainer for XGBoost and absent from every published `shap` wheel

### Fixed
- The `ModelSuite` example in the algorithm-selection guide set `calculate_shap: True`, contradicting the default and the config templates, and a suite is the most expensive place to enable it, since `SVM`, `KNN`, `GNB` and `MLP` route through `KernelExplainer`
- SHAP for tree models no longer converts the whole training set to pandas to build background data it never uses; the conversion is now made only by the explainers that need one
- The algorithm-selection guide put hyperparameters directly under the `model` step (`model: { learning_rate: 0.01 }`), where they are silently ignored; they belong under `model_params`. The suite example no longer tells users to give every method an empty entry, and both places now warn that a shared parameter must be one every listed model accepts
- A `model_params` section keyed by a model name was also handed to every other model, whose constructors rejected it (`unexpected keyword argument 'XGBoost'`), making per-model parameters unusable in `ModelSuite`. Unnamed models now receive only the shared parameters
- `MODEL_REGISTRY` aliased `SINGLE_MODEL_REGISTRY` instead of copying it, so importing it added `ModelSuite` to the single-model registry, letting a suite list itself among its own methods
- A non-mapping value under a model name now raises `ValueError` naming the model, instead of an unpacking `TypeError`

## [0.10.0] - 2026-08-11
### Added
- `read_config_template(stage, extension)` returns a built-in template as a configuration object, taking the same arguments as `write_config_template` but skipping the file. Useful for inspecting a stage's defaults (`print(aq.read_config_template("prepare"))`) and for building a configuration in code without one on disk
- `print(config)` now summarizes the configuration: source file, schema status, targets and their flag values, features (or NRT QC items), input file, active row filters, and the class and output directory of every step. Before an entry is selected it lists the available entry names. Also available as a string from `config.summary()`
- `ConfigBase.check_schema()` reports schema validity without setting `valid_yaml`, so reporting code does not change state under a caller

### Changed
- `repr(config)` now names the concrete config class and the selected entry, instead of always reporting `ConfigBase` and the section alone
- The built-in templates live in one registry shared by the config classes and the interface, so the default `prepare` template is now reachable as `template:data_sets_all`; previously it was the one variant `write_config_template` could write but no config class could load

## [0.9.1] - 2026-08-10
### Fixed
- Fresh installs failed while building `llvmlite 0.36.0`. `shap` requires `numba`, every `numba` supporting numpy 2.x caps numpy below the current release, so the resolver walked back to a `numba` with no wheel for supported Python versions. A `numba>=0.62` floor keeps resolution on wheels; this also fixes `uv sync` on Intel macOS, whose lockfile entry carried the same pin

## [0.9.0] - 2026-08-10
### Added
- The NRT QC guide lists all eleven QC items with what each one flags, grouped by profile- and observation-level
- New how-to page on using the QC items as model input features: configuration, the `params` / `col_names` / `fail_flag` settings, and the pitfalls: circularity when labelling from NRT flags, items that never fire, collinear columns, and flag values read as magnitudes by non-tree models

### Changed
- A training, validation, test or classification dataset with no rows now raises an error naming the target and the likely cause, instead of reaching the model and failing there as a feature-name mismatch. Splits are checked before any are written, so a failure leaves no partial output
- Fitting a model on single-class labels now raises an error naming the target; such a model predicts one class at one constant score that no `prediction_threshold` can separate. Evaluating against single-class labels stays a warning, and label-free classification is unaffected
- `pres` is no longer a target in the config templates or documentation examples: `pres_qc` rarely carries bad flags, so it trained a model that could flag nothing. Pressure remains an input feature and profile ordering column

### Fixed
- The `target_sets` reference had `pos_flag_values` / `neg_flag_values` described the wrong way round: the positive class is the bad observations (flagged 4, 6, 7), which is what the model detects

## [0.8.0] - 2026-08-10
### Fixed
- A leading `~` in a path is now expanded to the home directory, in `base_path` values read from a config file and in paths passed to `write_config_template`, `read_config`, `read_input_file`, `get_summary_stats` and `run_batch`. Previously `base_path: ~/aiqc_project/data` silently wrote every output into a literal `~` folder under the working directory

## [0.7.1] - 2026-08-09
### Fixed
- Documentation corrections: the tutorial chain skipped the input-data page, a broken cross-reference in the classification tutorial, a `classification_sets` example missing the required `feature_stats_set`, a non-existent `TimeSeriesValidation` class, wrong prediction column names and model output folder, and claims of hyperparameter tuning the library does not do
- Setup instructions: `uv sync` already installs the project, so the extra `uv pip install -e .` is gone; conda-forge install is marked as not yet published; the conda-forge version-bump steps in `RELEASING.md` targeted the wrong repository

## [0.7.0] - 2026-08-09
### Added
- `write_config_template(..., overwrite=True)` replaces an existing file
- `run_batch` runs `prepare` / `train` / `classify` (or `all`) over a table of dataset names, returning a per-run summary; `available_modes` lists the modes. Without a table each phase runs once, letting each config select its own set

### Changed
- `write_config_template` refuses to replace an existing file, raising `FileExistsError`; pass `overwrite=True` for the previous behaviour

### Fixed
- `read_config(file, set_name=...)` selects the named set from a file holding several sets; auto-selection ran first and rejected such a file, so `set_name` needed `auto_select=False` to work at all

## [0.6.0] - 2026-08-09
### Added
- `verbose=True` on `create_training_dataset`, `train_and_evaluate`, `run_nrt_qc` and `classify_dataset` prints each main step with the elapsed time
- `write_config_template(..., create_dirs=True)` creates a missing output directory; by default the refusal message names the option (and flags an unexpanded `~`)

### Changed
- Single-class evaluations now warn that their 1.0 scores are degenerate and name the target, replacing matplotlib's "No artists with labels found to put in legend"; the empty metric plot says why it is empty
- QC flag columns may be integer, string or float, and `pos_flag_values` / `neg_flag_values` may be written as `4` or `"4"`; the emitted `flag` column is now always Int64

### Fixed
- `k_fold` stays an integer column when a class holds fewer rows than `k_fold` (an empty numpy array defaulted to float64), fixing a `SchemaError` in the split step
- Summary stats accept `profile_no` in any integer dtype (previously Int32 only), so inputs from `ctddump` (UInt32) and auto-created identifier columns (Int64) no longer raise a `SchemaError`

## [0.5.0] - 2026-07-15
### Added
- NRT QC module: automated real-time QC tests (RTQC2/3/6/7/8/9/11/12/13/14 + temp-to-psal propagation), per-item flag columns, final NRT flags, and flag comparison reports (`run_nrt_qc`, `stage="nrt_qc"`)
- QC items as feature classes (`qc_*`), reusable in prepare feature sets
- EOS-80 seawater utilities (UNESCO 1983 sigma0) and QC flag helpers

## [0.4.0] - 2026-07-03
### Added
- `skip_evaluation` for classifying unlabeled data (optional/empty `flag`); skips label creation and performance evaluation

## [0.3.1] - 2026-07-03
### Fixed
- Packaging config ported to hatchling; CHANGELOG.md now bundled in the wheel

## [0.3.0] - 2026-05-23
### Added
- Configurable threshold for predicting labels
- Extra normalization methods (auto_min_max, standard)
- Automatic input-column validation and type correction
- Automatic creation of profile_no / observation_no
- SHAP score import utility (read_shap_scores)

### Changed
- Output format of model scores for performance evaluation

## [0.2.1] - 2026-05-19

### Changed
- Refactored all unit tests

## [0.2.0] - 2026-05-11

### Changed
- `LogisticRegression`: `penalty="l2"` → `l1_ratio=0` for sklearn 1.8.
- Test fixtures moved to GitHub release assets.
- Regenerated test models for sklearn 1.8 / current XGBoost.

### Added
- `scripts/fetch_test_data.sh` for contributors.
- `scripts/regenerate_test_models.py` for maintainers.

## [0.1.2] - 2026-05-08
### Fixed
- UV lock

## [0.1.1] - 2026-05-08
### Added
- Automatic publication to PyPI
- Automatic process for RTD
- Recipe for Anaconda

## [0.1.0] - 2026-05-08
### Added
- Port from dmqclib
