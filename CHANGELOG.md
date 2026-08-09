# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
As this project is still in active development, it does not yet strictly adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Changed
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
