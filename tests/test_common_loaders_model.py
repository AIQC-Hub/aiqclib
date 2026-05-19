"""Unit tests for the ``load_model_class`` loader (XGBoost + ModelSuite paths).

This file is the "outer-shell" companion to ``test_common_loaders_single_model.py``:
both exercise ``load_model_class``, but this file focuses on the default
XGBoost path and the ModelSuite wrapper, while the sibling file enumerates
all 9 single-model wrappers.

Coverage:
- Both ``"XGBoost"`` and the short ``"XGB"`` alias produce an XGBoost instance
- Both ``"ModelSuite"`` and the short ``"MS"`` alias produce a ModelSuite instance
- An invalid config string raises ValueError
- Calling build/predict/create_report on a freshly-loaded wrapper (no
  training_set, no test_set, no predictions) raises ValueError

Refactored from a ``unittest.TestCase`` class. The two "valid config" tests
collapse into one parametrize with 4 (config_name, expected_class) cases.
The 5 error tests stay as separate methods — each exercises a distinct
error path on ModelBase.

Note: the 5 error tests at the bottom are duplicated in
``test_common_loaders_single_model.py`` — both files call ``load_model_class``
with the default config (which yields XGBoost) and exercise the same error
paths. Preserving both for maximum regression coverage.
"""

import pytest

from aiqclib.common.loader.model_loader import load_model_class
from aiqclib.train.models.model_suite import ModelSuite
from aiqclib.train.models.xgboost import XGBoost


# ---------------------------------------------------------------------------
# Parametrize cases for the valid-config loader test
# ---------------------------------------------------------------------------

# Each row: (config string, expected wrapper class).
# Both long-form names and their short aliases should map to the same class.
_VALID_MODEL_NAMES = [
    ("XGBoost",    XGBoost),
    ("XGB",        XGBoost),
    ("ModelSuite", ModelSuite),
    ("MS",         ModelSuite),
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestModelClassLoader:
    """Tests for load_model_class's XGBoost + ModelSuite paths and error states."""

    @pytest.mark.parametrize(
        "config_name, expected_class",
        _VALID_MODEL_NAMES,
        ids=[case[0] for case in _VALID_MODEL_NAMES],
    )
    def test_load_model_class_by_name(
        self, config_name, expected_class, training_config_001,
    ):
        """Each (config_name, expected_class) pair produces the right wrapper.

        Verifies that load_model_class correctly maps both the long and short
        forms to the same underlying wrapper class.
        """
        training_config_001.data["step_class_set"]["steps"]["model"] = config_name
        ds = load_model_class(training_config_001)
        assert isinstance(ds, expected_class)

    def test_load_model_invalid_config(self, training_config_001):
        """An unrecognized model name in the config raises ValueError."""
        training_config_001.data["step_class_set"]["steps"]["model"] = "invalid_model_name"
        with pytest.raises(ValueError):
            _ = load_model_class(training_config_001)

    # ----- Error states on the loaded wrapper (default XGBoost) -----

    def test_build_model_empty_training_set(self, training_config_001):
        """build() with no training_set raises ValueError."""
        ds = load_model_class(training_config_001)
        with pytest.raises(ValueError):
            ds.build()

    def test_predict_model_empty_test_set(self, training_config_001):
        """predict() with no test_set raises ValueError."""
        ds = load_model_class(training_config_001)
        with pytest.raises(ValueError):
            ds.predict()

    def test_create_report_empty_test_set(self, training_config_001):
        """create_report() with no test_set raises ValueError."""
        ds = load_model_class(training_config_001)
        with pytest.raises(ValueError):
            ds.create_report()

    def test_create_report_empty_predictions(self, training_config_001):
        """create_report() with test_set set but no predictions raises ValueError.

        Sets test_set to ``{}`` to bypass the "test_set not set" check and
        force the path that validates ``predictions`` instead.
        """
        ds = load_model_class(training_config_001)
        ds.test_set = {}
        with pytest.raises(ValueError):
            ds.create_report()