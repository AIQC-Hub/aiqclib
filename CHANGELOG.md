# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
As this project is still in active development, it does not yet strictly adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
