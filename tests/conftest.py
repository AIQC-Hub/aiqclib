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

# Per-target list, used by every per-target loop in the suite.
TARGETS: tuple[str, ...] = ("temp", "psal", "pres")


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
    return _load_training_config("test_training_001.yaml")


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
# Training-input wiring — replicates the old setup_training_step2 /
# setup_training_step4 helpers
# ----------------------------------------------------------------------------

def _build_training_input(config: TrainingConfig, training_dir: Path):
    """Construct a ds_input with wiring identical to the legacy helper.

    Reads ``train_set_{target}.parquet`` and ``test_set_{target}.parquet`` for
    each of TARGETS from ``training_dir``, points the loader at them, and
    runs ``process_targets``.
    """
    input_file_names = {
        kind: {tgt: str(training_dir / f"{kind}_set_{tgt}.parquet") for tgt in TARGETS}
        for kind in ("train", "test")
    }
    ds_input = load_step1_input_training_set(config)
    ds_input.input_file_names = input_file_names
    ds_input.process_targets()
    return ds_input


@pytest.fixture
def training_input_001(training_config_001):
    """ds_input wired against tests/data/training/ for training config 001.

    Mirrors what the legacy ``setup_training_step2(test_obj)`` helper did:
    builds the loader, points it at the train/test parquets, and calls
    process_targets. Use this whenever a test starts with
    ``KFoldValidation(config, training_sets=ds_input.training_sets)``.
    """
    return _build_training_input(training_config_001, TRAINING_DIR)


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