"""Shared pytest fixtures for the aiqclib test suite.

This file centralises three kinds of setup that previously lived in every
``setUp``:

1. **Path constants** (``TESTS_DIR``, ``DATA_DIR``, etc.) — exposed both as
   module-level constants for import and as fixtures for use as test arguments.
2. **Config loaders** — one fixture per (stage, config number) pair, each
   returning a fresh, ``select()``-ed config object. Function-scoped, so
   mutations don't leak between tests.
3. **Common dataset/training input wiring** — fixtures that produce the
   ``ds_input`` object the way previous ``setup_training_step2`` /
   ``setup_training_step4`` module-level helpers did.

Conventions:
- Fixtures are snake_case.
- Configs are pre-selected with ``NRT_BO_001``. If a test needs the un-selected
  state, load via ``DataSetConfig(str(dataset_yaml_001))`` directly.
- Test output files go under ``DATA_DIR / "test"``. Use the ``test_output_dir``
  fixture and clean up manually with ``os.remove`` so commenting out the
  cleanup lets you inspect outputs after a failure.

When adding fixtures here, keep them general — anything used by only one or
two files should live in those files, not here.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Callable, List, Optional, Tuple

import pytest

from aiqclib.common.config.classify_config import ClassificationConfig
from aiqclib.common.config.dataset_config import DataSetConfig
from aiqclib.common.config.training_config import TrainingConfig
from aiqclib.common.loader.classify_loader import (
    load_classify_step1_input_dataset,
    load_classify_step2_summary_dataset,
    load_classify_step3_select_dataset,
    load_classify_step4_locate_dataset,
    load_classify_step5_extract_dataset,
)
from aiqclib.common.loader.dataset_loader import (
    load_step1_input_dataset,
    load_step2_summary_dataset,
    load_step3_select_dataset,
    load_step4_locate_dataset,
    load_step5_extract_dataset,
)
from aiqclib.common.loader.training_loader import load_step1_input_training_set


# ----------------------------------------------------------------------------
# Path constants (also exposed as fixtures further down)
# ----------------------------------------------------------------------------

TESTS_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = TESTS_DIR / "data"
CONFIG_DIR: Path = DATA_DIR / "config"
INPUT_DIR: Path = DATA_DIR / "input"
TRAINING_DIR: Path = DATA_DIR / "training"
NEGX5_TRAINING_DIR: Path = DATA_DIR / "negx5_training"
NEGX5_MODEL_DIR: Path = DATA_DIR / "negx5_model"
TEST_OUTPUT_DIR: Path = DATA_DIR / "test"

# All three QC targets the pipeline supports. Use this when iterating over
# targets in tests that exercise config, paths, identity, or anything where
# every target is structurally valid.
TARGETS: tuple[str, ...] = ("temp", "psal", "pres")

# Subset of TARGETS for which the reduced test fixtures contain usable data
# OR for which a test config genuinely excludes pres. The pres target is
# problematic for two reasons that may co-occur:
#   1. The reduced test fixtures have no ``pres_qc == 4`` rows in the test
#      split, so pres test data is empty.
#   2. Some test configs (e.g. ``NRT_BO_002`` in test_training_001.yaml)
#      use a 2-target ``target_set`` that excludes pres entirely.
#
# Tests that iterate over ``ds.training_sets`` / ``ds.test_sets`` /
# ``ds.models`` / ``ds.final_models`` / ``ds.output_file_names`` under
# either of those configurations must use this constant instead of TARGETS,
# because the per-target dict won't have a ``pres`` key.
#
# WHEN TO REMOVE: once the library handles zero-row test data gracefully
# (raising a clear error or skipping the empty target), the test fixtures
# are regenerated to include pres test rows, and tests are migrated back
# to a single 3-target config. At that point:
#   grep -rn "TARGETS_NONEMPTY" tests/    # gives the migration list
#   sed -i 's/TARGETS_NONEMPTY/TARGETS/g' tests/*.py
# then delete this constant.
TARGETS_NONEMPTY: tuple[str, ...] = ("temp", "psal")


# ----------------------------------------------------------------------------
# Path fixtures
# ----------------------------------------------------------------------------

@pytest.fixture
def tests_dir() -> Path:
    """Absolute path to the ``tests/`` directory."""
    return TESTS_DIR


@pytest.fixture
def data_dir() -> Path:
    """``tests/data/`` — root of all test fixtures."""
    return DATA_DIR


@pytest.fixture
def config_dir() -> Path:
    """``tests/data/config/`` — YAML config files."""
    return CONFIG_DIR


@pytest.fixture
def input_dir() -> Path:
    """``tests/data/input/`` — raw pre-pipeline input data."""
    return INPUT_DIR


@pytest.fixture
def training_dir() -> Path:
    """``tests/data/training/`` — train/test parquet files and model joblibs."""
    return TRAINING_DIR


@pytest.fixture
def test_output_dir() -> Path:
    """``tests/data/test/`` — destination for test-generated output files.

    Tests should write here, assert the file exists, then ``os.remove(...)``.
    Comment out the remove to inspect outputs after a failure.
    """
    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return TEST_OUTPUT_DIR


@pytest.fixture
def test_data_file() -> Path:
    """The primary input parquet used by most prepare-stage tests."""
    return INPUT_DIR / "nrt_cora_bo_test.parquet"


# ----------------------------------------------------------------------------
# YAML path fixtures (paths, not loaded configs — used by tests that exercise
# config loading itself, e.g. test_common_utils_config.py)
# ----------------------------------------------------------------------------

@pytest.fixture
def dataset_yaml_001() -> Path:
    return CONFIG_DIR / "test_dataset_001.yaml"


@pytest.fixture
def dataset_yaml_002() -> Path:
    return CONFIG_DIR / "test_dataset_002.yaml"


@pytest.fixture
def dataset_yaml_003() -> Path:
    return CONFIG_DIR / "test_dataset_003.yaml"


@pytest.fixture
def dataset_yaml_004() -> Path:
    return CONFIG_DIR / "test_dataset_004.yaml"


@pytest.fixture
def dataset_yaml_005() -> Path:
    return CONFIG_DIR / "test_dataset_005.yaml"


@pytest.fixture
def training_yaml_001() -> Path:
    return CONFIG_DIR / "test_training_001.yaml"


@pytest.fixture
def training_yaml_002() -> Path:
    return CONFIG_DIR / "test_training_002.yaml"


@pytest.fixture
def training_yaml_003() -> Path:
    return CONFIG_DIR / "test_training_003.yaml"


@pytest.fixture
def classify_yaml_001() -> Path:
    return CONFIG_DIR / "test_classify_001.yaml"


@pytest.fixture
def classify_yaml_002() -> Path:
    return CONFIG_DIR / "test_classify_002.yaml"


@pytest.fixture
def classify_yaml_003() -> Path:
    return CONFIG_DIR / "test_classify_003.yaml"


# ----------------------------------------------------------------------------
# Config loaders — pre-selected, function-scoped (fresh per test, no mutation
# leakage)
# ----------------------------------------------------------------------------

_DEFAULT_SELECT: str = "NRT_BO_001"


def _load_dataset_config(filename: str, select: str = _DEFAULT_SELECT) -> DataSetConfig:
    """Load and select a DataSetConfig from CONFIG_DIR/filename."""
    config = DataSetConfig(str(CONFIG_DIR / filename))
    config.select(select)
    return config


def _load_training_config(filename: str, select: str = _DEFAULT_SELECT) -> TrainingConfig:
    """Load and select a TrainingConfig from CONFIG_DIR/filename."""
    config = TrainingConfig(str(CONFIG_DIR / filename))
    config.select(select)
    return config


def _load_classify_config(filename: str, select: str = _DEFAULT_SELECT) -> ClassificationConfig:
    """Load and select a ClassificationConfig from CONFIG_DIR/filename."""
    config = ClassificationConfig(str(CONFIG_DIR / filename))
    config.select(select)
    return config


@pytest.fixture
def dataset_config_001() -> DataSetConfig:
    return _load_dataset_config("test_dataset_001.yaml")


@pytest.fixture
def dataset_config_002() -> DataSetConfig:
    return _load_dataset_config("test_dataset_002.yaml")


@pytest.fixture
def dataset_config_003() -> DataSetConfig:
    return _load_dataset_config("test_dataset_003.yaml")


@pytest.fixture
def dataset_config_004() -> DataSetConfig:
    return _load_dataset_config("test_dataset_004.yaml")


@pytest.fixture
def dataset_config_005() -> DataSetConfig:
    return _load_dataset_config("test_dataset_005.yaml")


@pytest.fixture
def training_config_001() -> TrainingConfig:
    """Selects ``NRT_BO_001`` — 3-target (temp, psal, pres)."""
    return _load_training_config("test_training_001.yaml")


@pytest.fixture
def training_config_001_bo002() -> TrainingConfig:
    """test_training_001.yaml with NRT_BO_002 selected — 2-target (temp, psal).

    NRT_BO_002 uses target_set_1_2 which excludes pres. Useful for tests
    that exercise build/test pipelines where the reduced fixtures have zero
    rows for pres in the test split. Same YAML, same path_info, different
    target_set.
    """
    return _load_training_config("test_training_001.yaml", select="NRT_BO_002")


@pytest.fixture
def training_config_002() -> TrainingConfig:
    return _load_training_config("test_training_002.yaml")


@pytest.fixture
def training_config_003() -> TrainingConfig:
    return _load_training_config("test_training_003.yaml")


@pytest.fixture
def classify_config_001() -> ClassificationConfig:
    return _load_classify_config("test_classify_001.yaml")


@pytest.fixture
def classify_config_002() -> ClassificationConfig:
    return _load_classify_config("test_classify_002.yaml")


@pytest.fixture
def classify_config_003() -> ClassificationConfig:
    return _load_classify_config("test_classify_003.yaml")


# ----------------------------------------------------------------------------
# Dataset-pipeline wiring (prepare-stage step1 + step3 outputs)
#
# These fixtures mirror the legacy setUp pattern: load step1 input dataset
# against the test parquet, optionally chain into step3 select. Used by
# tests for prepare steps 2-6, each of which depends on outputs from earlier
# steps in the same pipeline.
# ----------------------------------------------------------------------------

def _build_dataset_input(config: DataSetConfig, test_data_file: Path):
    """Construct a step1 input dataset and read the input parquet.

    Returns the populated ds_input. The legacy setUp in every prepare-stage
    test does this exact sequence; here it lives once.
    """
    ds = load_step1_input_dataset(config)
    ds.input_file_name = str(test_data_file)
    ds.read_input_data()
    return ds


def _build_dataset_select(config: DataSetConfig, input_data):
    """Construct a step3 select dataset and run label_profiles().

    Returns the populated ds_select with ``selected_profiles`` ready to use
    downstream. Step4+ tests typically need this.
    """
    ds = load_step3_select_dataset(config, input_data=input_data)
    ds.label_profiles()
    return ds


@pytest.fixture
def dataset_input_001(dataset_config_001, test_data_file):
    """step1 input dataset for test_dataset_001.yaml."""
    return _build_dataset_input(dataset_config_001, test_data_file)


@pytest.fixture
def dataset_input_003(dataset_config_003, test_data_file):
    """step1 input dataset for test_dataset_003.yaml (NegX5 variant)."""
    return _build_dataset_input(dataset_config_003, test_data_file)


@pytest.fixture
def dataset_input_004(dataset_config_004, test_data_file):
    """step1 input dataset for test_dataset_004.yaml."""
    return _build_dataset_input(dataset_config_004, test_data_file)


@pytest.fixture
def dataset_select_001(dataset_config_001, dataset_input_001):
    """step3 select dataset (labelled profiles) for test_dataset_001.yaml."""
    return _build_dataset_select(dataset_config_001, dataset_input_001.input_data)


@pytest.fixture
def dataset_select_003(dataset_config_003, dataset_input_003):
    """step3 select dataset (labelled profiles) for test_dataset_003.yaml (NegX5)."""
    return _build_dataset_select(dataset_config_003, dataset_input_003.input_data)


# ----------------------------------------------------------------------------
# Full-pipeline helper for prepare-stage tests
#
# build_prepare_pipeline mirrors the classify-side run_classify_prepare_pipeline
# helper. Used by prepare-stage tests (step5_extract, step6_split, etc.) that
# need multiple upstream outputs in the same test. ``stop_after`` lets a test
# avoid running unneeded later stages.
# ----------------------------------------------------------------------------

_STAGES_ORDER: tuple[str, ...] = ("input", "summary", "select", "locate", "extract")


def build_prepare_pipeline(
    config: DataSetConfig,
    test_data_file: Path,
    *,
    stop_after: str = "extract",
) -> SimpleNamespace:
    """Run the prepare pipeline through the chosen stage.

    Returns a SimpleNamespace with attributes for each stage that ran:
    ``config`` (always), and any of ``input``, ``summary``, ``select``,
    ``locate``, ``extract``. ``stop_after`` controls how far through the
    pipeline to run — use the earliest stage your test actually needs to
    keep test time down.

    :param config: a fresh, select()-ed DataSetConfig
    :param test_data_file: path to the input parquet
    :param stop_after: one of "input", "summary", "select", "locate", "extract"
    :returns: SimpleNamespace with config + per-stage output datasets
    """
    if stop_after not in _STAGES_ORDER:
        raise ValueError(
            f"stop_after must be one of {_STAGES_ORDER}, got {stop_after!r}"
        )
    final_idx = _STAGES_ORDER.index(stop_after)

    result = SimpleNamespace(config=config)

    ds_input = load_step1_input_dataset(config)
    ds_input.input_file_name = str(test_data_file)
    ds_input.read_input_data()
    result.input = ds_input
    if final_idx == _STAGES_ORDER.index("input"):
        return result

    ds_summary = load_step2_summary_dataset(config, input_data=ds_input.input_data)
    ds_summary.calculate_stats()
    result.summary = ds_summary
    if final_idx == _STAGES_ORDER.index("summary"):
        return result

    ds_select = load_step3_select_dataset(config, input_data=ds_input.input_data)
    ds_select.label_profiles()
    result.select = ds_select
    if final_idx == _STAGES_ORDER.index("select"):
        return result

    ds_locate = load_step4_locate_dataset(
        config,
        input_data=ds_input.input_data,
        selected_profiles=ds_select.selected_profiles,
    )
    ds_locate.process_targets()
    result.locate = ds_locate
    if final_idx == _STAGES_ORDER.index("locate"):
        return result

    ds_extract = load_step5_extract_dataset(
        config,
        input_data=ds_input.input_data,
        selected_profiles=ds_select.selected_profiles,
        selected_rows=ds_locate.selected_rows,
        summary_stats=ds_summary.summary_stats,
    )
    ds_extract.process_targets()
    result.extract = ds_extract
    return result


# ----------------------------------------------------------------------------
# Training-input wiring — replicates the old setup_training_step2 /
# setup_training_step4 helpers
# ----------------------------------------------------------------------------

def _build_training_input(
    config: TrainingConfig, training_dir: Path, targets: tuple[str, ...] = TARGETS
):
    """Construct a ds_input with wiring identical to the legacy helper.

    Reads ``train_set_{target}.parquet`` and ``test_set_{target}.parquet`` for
    each target in ``targets`` from ``training_dir``, points the loader at
    them, and runs ``process_targets``.

    :param targets: which targets to load files for. Defaults to TARGETS;
        pass TARGETS_NONEMPTY for configs that exclude pres.
    """
    input_file_names = {
        kind: {tgt: str(training_dir / f"{kind}_set_{tgt}.parquet") for tgt in targets}
        for kind in ("train", "test")
    }
    ds_input = load_step1_input_training_set(config)
    ds_input.input_file_names = input_file_names
    ds_input.process_targets()
    return ds_input


@pytest.fixture
def training_input_001(training_config_001):
    """ds_input wired against tests/data/training/ for training config 001.

    Uses NRT_BO_001 (3-target). Mirrors what the legacy
    ``setup_training_step2(test_obj)`` helper did: builds the loader, points
    it at the train/test parquets, and calls process_targets. Use this
    whenever a test starts with
    ``KFoldValidation(config, training_sets=ds_input.training_sets)``.
    """
    return _build_training_input(training_config_001, TRAINING_DIR)


@pytest.fixture
def training_input_001_bo002(training_config_001_bo002):
    """ds_input wired against tests/data/training/ for training config 001 / NRT_BO_002.

    Like ``training_input_001`` but loads only temp + psal parquets (the
    2-target target_set under NRT_BO_002). Use this whenever a test would
    fail with ``training_input_001`` because pres test data is empty.
    """
    return _build_training_input(
        training_config_001_bo002, TRAINING_DIR, targets=TARGETS_NONEMPTY
    )


@pytest.fixture
def training_input_negx5(training_config_003):
    """ds_input wired against tests/data/negx5_training/ for training config 003."""
    return _build_training_input(training_config_003, NEGX5_TRAINING_DIR)


# ----------------------------------------------------------------------------
# Classify-pipeline helper — runs prepare steps 1-5 against one or more
# classify configs and returns the paired (configs, extracts).
#
# Used by step6 (classify_all + classify_suite) and may be reused by future
# step7 (concat) tests. Lives in conftest because two+ files use it.
# ----------------------------------------------------------------------------

def run_classify_prepare_pipeline(
    config_files: List[Path],
    test_data_file: Path,
    mutate_config: Optional[Callable[[ClassificationConfig], None]] = None,
) -> Tuple[List[ClassificationConfig], List]:
    """Run prepare steps 1-5 for each config; return (configs, extracts).

    For each config file: load, select NRT_BO_001, optionally apply
    ``mutate_config(config)`` (e.g. to inject ModelSuite settings), then run
    the five prepare steps in order. The returned ``extracts[idx]`` is the
    ``ds_extract`` whose ``target_features`` is the test set ClassifyAll(Suite)
    consumes.

    :param config_files: list of YAML paths, one per config to test
    :param test_data_file: path to the input parquet (typically
        ``nrt_cora_bo_test.parquet``)
    :param mutate_config: optional callback applied to each config after
        select() but before the pipeline runs. Used by suite tests to inject
        ``ClassifyAllSuite`` / ``ModelSuite`` settings.
    :returns: ``(configs, extracts)`` lists in the input order.
    """
    configs = []
    extracts = []
    for path in config_files:
        config = ClassificationConfig(str(path))
        config.select(_DEFAULT_SELECT)
        if mutate_config is not None:
            mutate_config(config)

        ds_input = load_classify_step1_input_dataset(config)
        ds_input.input_file_name = str(test_data_file)
        ds_input.read_input_data()

        ds_summary = load_classify_step2_summary_dataset(
            config, input_data=ds_input.input_data
        )
        ds_summary.calculate_stats()

        ds_select = load_classify_step3_select_dataset(
            config, input_data=ds_input.input_data
        )
        ds_select.label_profiles()

        ds_locate = load_classify_step4_locate_dataset(
            config,
            input_data=ds_input.input_data,
            selected_profiles=ds_select.selected_profiles,
        )
        ds_locate.process_targets()

        ds_extract = load_classify_step5_extract_dataset(
            config,
            input_data=ds_input.input_data,
            selected_profiles=ds_select.selected_profiles,
            selected_rows=ds_locate.selected_rows,
            summary_stats=ds_summary.summary_stats,
        )
        ds_extract.process_targets()

        configs.append(config)
        extracts.append(ds_extract)

    return configs, extracts