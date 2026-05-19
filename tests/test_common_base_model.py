"""Unit tests for the ``ModelBase`` class.

Coverage:
- A subclass with no ``expected_class_name`` raises NotImplementedError on
  construction
- A subclass whose ``expected_class_name`` doesn't match the config-selected
  model class raises ValueError
- ``__str__`` returns a structured representation
- ``load_model`` raises FileNotFoundError on missing path and ValueError on
  type mismatch between the loaded joblib and ``_get_model_class()``
- The SHAP flag (``calculate_shap``) propagates from config to ``enable_shap``
- ``update_contingency_table`` validates required member variables and
  correctly accumulates per-fold contingency rows

Refactored from a ``unittest.TestCase`` class. The three module-level mock
subclasses (ModelBaseWithEmptyName, ModelBaseWithExpectedName,
ModelBaseWithWrongName) stay at module level.

Fix to original: the file's top-level docstring claimed this tested
"DataSetBase in aiqclib.common.base.model_base" — a copy-paste error.
This file tests ``ModelBase`` (which lives in ``aiqclib.common.base.model_base``).
"""

from typing import Self

import polars as pl
import pytest
import xgboost as xgb

from aiqclib.common.base.config_base import ConfigBase
from aiqclib.common.base.model_base import ModelBase


# ---------------------------------------------------------------------------
# Module-level mock subclasses
# ---------------------------------------------------------------------------

class ModelBaseWithEmptyName(ModelBase):
    """Subclass with no ``expected_class_name`` — used to test the
    NotImplementedError path in ModelBase's constructor."""

    def __init__(self, config: ConfigBase) -> None:
        super().__init__(config)

    def build(self) -> None:
        pass

    def test(self) -> None:
        pass

    def update_nthreads(self, model: Self) -> Self:
        return model

    def _get_model_class(self):
        pass


class ModelBaseWithExpectedName(ModelBase):
    """Subclass whose ``expected_class_name`` matches the config's model
    step ("XGBoost"). Used to test the successful-construction and
    type-checking paths."""

    expected_class_name: str = "XGBoost"

    def __init__(self, config: ConfigBase) -> None:
        super().__init__(config)

    def build(self) -> None:
        pass

    def test(self) -> None:
        pass

    def update_nthreads(self, model: Self) -> Self:
        return model

    def _get_model_class(self):
        return xgb.XGBClassifier


class ModelBaseWithWrongName(ModelBase):
    """Subclass whose ``expected_class_name`` ("XGBoostZ") doesn't match
    any registered model class — triggers the ValueError path."""

    expected_class_name: str = "XGBoostZ"

    def __init__(self, config: ConfigBase) -> None:
        super().__init__(config)

    def build(self) -> None:
        pass

    def test(self) -> None:
        pass

    def update_nthreads(self, model: Self) -> Self:
        return model

    def _get_model_class(self):
        return xgb.XGBClassifier


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestModelBaseMethods:
    """Tests for ModelBase's abstract-class behaviour, model loading, and
    contingency-table accumulation."""

    # ----- Identity / construction -----

    def test_expected_class_name(self, training_config_001):
        """A subclass without ``expected_class_name`` raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            _ = ModelBaseWithEmptyName(training_config_001)

    def test_model_name(self, training_config_001):
        """A subclass with a mismatched ``expected_class_name`` raises ValueError."""
        with pytest.raises(ValueError):
            _ = ModelBaseWithWrongName(training_config_001)

    def test_representing_str(self, training_config_001):
        """__str__ returns "ModelBase(class=<expected_class_name>)"."""
        ds = ModelBaseWithExpectedName(training_config_001)
        assert str(ds) == "ModelBase(class=XGBoost)"

    # ----- load_model -----

    def test_load_input_with_invalid_path(self, training_config_001):
        """load_model with a missing path raises FileNotFoundError."""
        ds = ModelBaseWithExpectedName(training_config_001)
        with pytest.raises(FileNotFoundError):
            ds.load_model("invalid_file_path")

    def test_load_model_success(self, training_config_001, training_dir):
        """load_model loads a joblib whose class matches ``_get_model_class()``.

        Uses ``model_temp_xgb.joblib`` — a temp-target XGBoost fixture.
        """
        ds = ModelBaseWithExpectedName(training_config_001)
        ds.load_model(str(training_dir / "model_temp_xgb.joblib"))
        assert isinstance(ds.model, xgb.XGBClassifier)

    def test_load_model_type_mismatch(self, training_config_001, training_dir):
        """load_model raises ValueError when the joblib's class doesn't match.

        Uses ``model_temp_mlp.joblib`` (an MLP model) loaded into a wrapper
        whose ``_get_model_class()`` returns XGBClassifier. ModelBase should
        catch the mismatch and surface a "Inconsistent class instances" error.
        """
        ds = ModelBaseWithExpectedName(training_config_001)
        with pytest.raises(ValueError, match="Inconsistent class instances"):
            ds.load_model(str(training_dir / "model_temp_mlp.joblib"))

    # ----- SHAP flag -----

    def test_shap_flag(self, training_config_001):
        """``calculate_shap`` in config propagates to ``enable_shap`` on the model.

        Unset == False; True propagates as True; explicit False propagates as False.
        Contrast with KFoldValidationSuite, which suppresses SHAP regardless
        of config — that override happens at the step level, not on
        ModelBase itself.
        """
        model = ModelBaseWithExpectedName(training_config_001)
        assert model.enable_shap is False  # unset == False

        training_config_001.data["step_param_set"]["steps"]["model"]["calculate_shap"] = True
        model = ModelBaseWithExpectedName(training_config_001)
        assert model.enable_shap is True

        training_config_001.data["step_param_set"]["steps"]["model"]["calculate_shap"] = False
        model = ModelBaseWithExpectedName(training_config_001)
        assert model.enable_shap is False

    # ----- update_contingency_table -----

    def test_update_contingency_table_validation(self, training_config_001):
        """update_contingency_table raises ValueError when required vars are missing.

        Two cases:
        1. test_set is None (with predictions set) → "Member variable 'test_set'"
        2. predictions is None (with test_set set) → "Member variable 'predictions'"
        """
        model = ModelBaseWithExpectedName(training_config_001)

        # Case 1: test_set is None
        model.test_set = None
        model.predictions = pl.DataFrame({"score": [0.5]})
        with pytest.raises(ValueError, match="Member variable 'test_set'"):
            model.update_contingency_table()

        # Case 2: predictions is None
        model.test_set = pl.DataFrame({"label": [1]})
        model.predictions = None
        with pytest.raises(ValueError, match="Member variable 'predictions'"):
            model.update_contingency_table()

    def test_update_contingency_table_flow(self, training_config_001):
        """Multi-batch contingency-table updates correctly initialize then append.

        Verifies the table is initialized on the first call (k=0) with the
        expected shape (3, 4) and columns, and that a subsequent k=1 call
        appends rows correctly to give a (5, 4) total.
        """
        model = ModelBaseWithExpectedName(training_config_001)

        # ----- Batch 1 (fold k=0) -----
        model.k = 0
        model.test_set = pl.DataFrame({"label": [0, 1, 0]})
        model.predictions = pl.DataFrame({
            "label": [0, 1, 0],
            "predicted_label": [0, 1, 0],
            "score": [0.1, 0.9, 0.4],
        })

        model.update_contingency_table()

        assert model.contingency_table is not None
        assert model.contingency_table.shape == (3, 4)
        assert model.contingency_table.columns == [
            "k", "label", "predicted_label", "score",
        ]

        # Verify Batch 1 content equals the expected frame exactly.
        expected_batch_1 = pl.DataFrame({
            "k": [0, 0, 0],
            "label": [0, 1, 0],
            "predicted_label": [0, 1, 0],
            "score": [0.1, 0.9, 0.4],
        })
        assert model.contingency_table.equals(expected_batch_1)

        # ----- Batch 2 (fold k=1) -----
        model.k = 1
        model.test_set = pl.DataFrame({"label": [1, 1]})
        model.predictions = pl.DataFrame({
            "label": [1, 0],
            "predicted_label": [1, 0],
            "score": [0.8, 0.3],
        })

        model.update_contingency_table()

        # Total rows now 3 + 2 = 5.
        assert model.contingency_table.shape == (5, 4)

        # k=1 rows specifically: 2 rows with the new scores.
        k1_rows = model.contingency_table.filter(pl.col("k") == 1)
        assert k1_rows.shape == (2, 4)
        assert k1_rows["score"].to_list() == [0.8, 0.3]