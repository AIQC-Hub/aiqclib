"""Unit tests for the training-stage loader functions.

Three classes cover the three loader functions:
- ``TestTrainingInputClassLoader``    — load_step1_input_training_set
- ``TestModelValidationClassLoader``  — load_step2_model_validation_class
- ``TestBuildModelClassLoader``       — load_step4_build_model_class

Each verifies that:
1. The loader returns an instance of the expected wrapper class with the
   right ``step_name``
2. An invalid step-class name raises ValueError
3. Provided training_sets / test_sets are propagated to the resulting
   wrapper with expected shapes

Refactored from ``unittest.TestCase`` classes. The setUp pattern that
wired ds_input via load_step1_input_training_set + process_targets is
replaced with the conftest ``training_input_001`` fixture (or
``training_input_001_bo002`` for the build tests, where pres test data
is empty).

Config choice per class:
- ``TestTrainingInputClassLoader``     uses ``training_config_001`` (3-target)
- ``TestModelValidationClassLoader``   uses ``training_config_001`` + ``training_input_001``
  (3-target — only checks training data, where pres is non-empty)
- ``TestBuildModelClassLoader``        uses ``training_config_001_bo002`` +
  ``training_input_001_bo002`` (2-target — checks test data, where pres
  is empty under the reduced fixtures)

When the library handles zero-row test data, switch ``TestBuildModelClassLoader``
back to ``training_config_001`` + ``training_input_001`` and use TARGETS
instead of TARGETS_NONEMPTY.
"""

import polars as pl
import pytest

from aiqclib.common.loader.training_loader import (
    load_step1_input_training_set,
    load_step2_model_validation_class,
    load_step4_build_model_class,
)
from aiqclib.train.step1_read_input.dataset_a import InputTrainingSetA
from aiqclib.train.step2_validate_model.kfold_validation import KFoldValidation
from aiqclib.train.step2_validate_model.kfold_validation_suite import (
    KFoldValidationSuite,
)
from aiqclib.train.step4_build_model.build_model import BuildModel
from aiqclib.train.step4_build_model.build_model_suite import BuildModelSuite

from tests.conftest import TARGETS, TARGETS_NONEMPTY


# ---------------------------------------------------------------------------
# Step 1: input
# ---------------------------------------------------------------------------


class TestTrainingInputClassLoader:
    """Tests for load_step1_input_training_set."""

    def test_load_dataset_valid_config(self, training_config_001):
        """Default config produces an InputTrainingSetA with step_name='input'."""
        ds = load_step1_input_training_set(training_config_001)
        assert isinstance(ds, InputTrainingSetA)
        assert ds.step_name == "input"

    def test_load_input_class_with_invalid_config(self, training_config_001):
        """An invalid input-class name raises ValueError."""
        training_config_001.data["step_class_set"]["steps"]["input"] = "InvalidClass"
        with pytest.raises(ValueError):
            _ = load_step1_input_training_set(training_config_001)


# ---------------------------------------------------------------------------
# Step 2: validation
# ---------------------------------------------------------------------------


class TestModelValidationClassLoader:
    """Tests for load_step2_model_validation_class.

    Validation only consumes training data. Pres training data is non-empty
    in the reduced fixtures (only pres TEST data is empty), so 3-target
    config + training_input_001 works cleanly here.
    """

    def test_load_dataset_valid_config(self, training_config_001):
        """Default config produces a KFoldValidation with step_name='validate'."""
        ds = load_step2_model_validation_class(training_config_001)
        assert isinstance(ds, KFoldValidation)
        assert ds.step_name == "validate"

    def test_load_validation_suite_dataset(self, training_config_001):
        """Suite config produces a KFoldValidationSuite with step_name='validate'."""
        training_config_001.data["step_class_set"]["steps"]["model"] = "ModelSuite"
        training_config_001.data["step_class_set"]["steps"]["validate"] = (
            "KFoldValidationSuite"
        )
        ds = load_step2_model_validation_class(training_config_001)
        assert isinstance(ds, KFoldValidationSuite)
        assert ds.step_name == "validate"

    def test_training_set_data(self, training_config_001, training_input_001):
        """Provided training_sets propagate to the KFoldValidation instance.

        Per-target shapes follow ``training_input_001`` (3-target). All
        three targets have non-empty training data.
        """
        ds = load_step2_model_validation_class(
            training_config_001,
            training_input_001.training_sets,
        )

        expected_rows = {"temp": 22, "psal": 34, "pres": 18}
        for tgt in TARGETS:
            assert isinstance(ds.training_sets[tgt], pl.DataFrame)
            assert ds.training_sets[tgt].shape[0] == expected_rows[tgt]
            assert ds.training_sets[tgt].shape[1] == 57

    def test_suit_training_set_data(self, training_config_001, training_input_001):
        """Same as test_training_set_data but for the suite variant."""
        training_config_001.data["step_class_set"]["steps"]["model"] = "ModelSuite"
        training_config_001.data["step_class_set"]["steps"]["validate"] = (
            "KFoldValidationSuite"
        )
        ds = load_step2_model_validation_class(
            training_config_001,
            training_input_001.training_sets,
        )

        expected_rows = {"temp": 22, "psal": 34, "pres": 18}
        for tgt in TARGETS:
            assert isinstance(ds.training_sets[tgt], pl.DataFrame)
            assert ds.training_sets[tgt].shape[0] == expected_rows[tgt]
            assert ds.training_sets[tgt].shape[1] == 57


# ---------------------------------------------------------------------------
# Step 4: build
# ---------------------------------------------------------------------------


class TestBuildModelClassLoader:
    """Tests for load_step4_build_model_class.

    Uses the bo002 fixtures (2-target NRT_BO_002) because the original
    test_training_and_test_sets asserts ``ds.test_sets["pres"].shape[0] == 12``,
    which fails under the reduced fixtures where pres test data has zero
    rows. NRT_BO_002 excludes pres entirely, so the per-target dicts only
    have temp + psal keys — iteration uses TARGETS_NONEMPTY.
    """

    def test_load_dataset_valid_config(self, training_config_001_bo002):
        """Default config produces a BuildModel with step_name='build'."""
        ds = load_step4_build_model_class(training_config_001_bo002)
        assert isinstance(ds, BuildModel)
        assert ds.step_name == "build"

    def test_load_validation_suite_dataset(self, training_config_001_bo002):
        """Suite config produces a BuildModelSuite with step_name='build'."""
        training_config_001_bo002.data["step_class_set"]["steps"]["model"] = (
            "ModelSuite"
        )
        training_config_001_bo002.data["step_class_set"]["steps"]["build"] = (
            "BuildModelSuite"
        )
        ds = load_step4_build_model_class(training_config_001_bo002)
        assert isinstance(ds, BuildModelSuite)
        assert ds.step_name == "build"

    def test_training_and_test_sets(
        self,
        training_config_001_bo002,
        training_input_001_bo002,
    ):
        """Provided training_sets + test_sets propagate to BuildModel."""
        ds = load_step4_build_model_class(
            training_config_001_bo002,
            training_input_001_bo002.training_sets,
            training_input_001_bo002.test_sets,
        )

        expected_train_rows = {"temp": 22, "psal": 34}
        expected_test_rows = {"temp": 2, "psal": 2}
        for tgt in TARGETS_NONEMPTY:
            assert isinstance(ds.training_sets[tgt], pl.DataFrame)
            assert ds.training_sets[tgt].shape[0] == expected_train_rows[tgt]
            assert ds.training_sets[tgt].shape[1] == 57

            assert isinstance(ds.test_sets[tgt], pl.DataFrame)
            assert ds.test_sets[tgt].shape[0] == expected_test_rows[tgt]
            assert ds.test_sets[tgt].shape[1] == 56

    def test_suit_training_and_test_sets(
        self,
        training_config_001_bo002,
        training_input_001_bo002,
    ):
        """Same as test_training_and_test_sets but for the suite variant."""
        training_config_001_bo002.data["step_class_set"]["steps"]["model"] = (
            "ModelSuite"
        )
        training_config_001_bo002.data["step_class_set"]["steps"]["build"] = (
            "BuildModelSuite"
        )
        ds = load_step4_build_model_class(
            training_config_001_bo002,
            training_input_001_bo002.training_sets,
            training_input_001_bo002.test_sets,
        )

        expected_train_rows = {"temp": 22, "psal": 34}
        expected_test_rows = {"temp": 2, "psal": 2}
        for tgt in TARGETS_NONEMPTY:
            assert isinstance(ds.training_sets[tgt], pl.DataFrame)
            assert ds.training_sets[tgt].shape[0] == expected_train_rows[tgt]
            assert ds.training_sets[tgt].shape[1] == 57

            assert isinstance(ds.test_sets[tgt], pl.DataFrame)
            assert ds.test_sets[tgt].shape[0] == expected_test_rows[tgt]
            assert ds.test_sets[tgt].shape[1] == 56
