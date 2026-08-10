# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
As this project is still in active development, it does not yet strictly adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- The NRT QC guide lists all eleven QC items with what each one flags, grouped by profile- and observation-level
- New how-to page on using the QC items as model input features, covering the `params` / `col_names` / `fail_flag` settings and the two items that need extra care

### Changed
- A training, validation, test or classification dataset with no rows now raises an error naming the target and the likely cause, instead of reaching the model and failing there as a feature-name mismatch. Splits are checked before any are written, so a failure leaves no partial output
- Fitting a model on single-class labels now raises an error naming the target; such a model predicts one class at one constant score that no `prediction_threshold` can separate. Evaluating against single-class labels stays a warning, and label-free classification is unaffected
- `pres` is no longer a target in the config templates or documentation examples: `pres_qc` rarely carries bad flags, so it trained a model that could flag nothing. Pressure remains an input feature and profile ordering column

### Fixed
- The `target_sets` reference had `pos_flag_values` / `neg_flag_values` described the wrong way round — the positive class is the bad observations (flagged 4, 6, 7), which is what the model detects

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
