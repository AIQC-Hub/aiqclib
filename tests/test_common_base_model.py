"""
Unit tests for the DataSetBase class in aiqclib.common.base.model_base
This module verifies the correct functionality of DataSetBase's methods.
"""

import unittest
from pathlib import Path
from typing import Self

import polars as pl
import xgboost as xgb

from aiqclib.common.base.config_base import ConfigBase
from aiqclib.common.base.model_base import ModelBase
from aiqclib.common.config.training_config import TrainingConfig


class ModelBaseWithEmptyName(ModelBase):
    """
    ModelBaseWithEmptyName is used to test methods and variables in ModelBase
    """

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
    """
    ModelBaseWithExpectedName is used to test methods and variables in ModelBase
    """

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
    """
    ModelBaseWithWrongName is used to test methods and variables in ModelBase
    """

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


class TestModelBaseMethods(unittest.TestCase):
    """
    A suite of tests that verify the correctness of methods
    within the ModelBase.
    """

    def setUp(self):
        """
        Set up a reference to the test configuration file (test_training_001.yaml)
        to be used by all subsequent tests in this class.
        """
        self.config_file_path = (
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_training_001.yaml"
        )
        self.config = TrainingConfig(str(self.config_file_path))
        self.config.select("NRT_BO_001")

    def test_expected_class_name(self):
        """
        Ensure that an undefined expected class_name raises a NotImplementedError.
        """
        with self.assertRaises(NotImplementedError):
            _ = ModelBaseWithEmptyName(self.config)

    def test_model_name(self):
        """
        Ensure that an unmatched model name raises a ValueError.
        """
        with self.assertRaises(ValueError):
            _ = ModelBaseWithWrongName(self.config)

    def test_representing_str(self):
        """
        Ensure that the instance returns a correct string representation.
        """
        ds = ModelBaseWithExpectedName(self.config)
        self.assertEqual(str(ds), "ModelBase(class=XGBoost)")

    def test_load_input_with_invalid_path(self):
        """
        Ensure that an invalid file path raises a FileNotFoundError.
        """
        ds = ModelBaseWithExpectedName(self.config)
        with self.assertRaises(FileNotFoundError):
            ds.load_model("invalid_file_path")

    def test_load_model_success(self):
        """
        Ensure that load_model successfully loads a model when the class matches
        the one expected by _get_model_class().
        """
        ds = ModelBaseWithExpectedName(self.config)

        model_file = (
            Path(__file__).resolve().parent
            / "data"
            / "training"
            / "model_pres_xgb.joblib"
        )

        ds.load_model(str(model_file))
        self.assertIsInstance(ds.model, xgb.XGBClassifier)

    def test_load_model_type_mismatch(self):
        """
        Ensure that load_model raises a ValueError when the loaded model
        does not match the expected class.
        """
        ds = ModelBaseWithExpectedName(self.config)

        model_file = (
            Path(__file__).resolve().parent
            / "data"
            / "training"
            / "model_pres_mlp.joblib"
        )

        # Verify that loading the wrong model type triggers the custom ValueError
        with self.assertRaisesRegex(ValueError, "Inconsistent class instances"):
            ds.load_model(str(model_file))

    def test_update_contingency_table_validation(self):
        """
        Ensure that update_contingency_table raises ValueError when required
        member variables are missing.
        """
        model = ModelBaseWithExpectedName(self.config)

        # Case 1: test_set is None
        model.test_set = None
        model.predictions = pl.DataFrame({"score": [0.5]})
        with self.assertRaisesRegex(ValueError, "Member variable 'test_set'"):
            model.update_contingency_table()

        # Case 2: predictions is None
        model.test_set = pl.DataFrame({"label": [1]})
        model.predictions = None
        with self.assertRaisesRegex(ValueError, "Member variable 'predictions'"):
            model.update_contingency_table()

    def test_shap_flag(self):
        model = ModelBaseWithExpectedName(self.config)
        self.assertFalse(model.enable_shap)

        self.config.data["step_param_set"]["steps"]["model"]["calculate_shap"] = True
        model = ModelBaseWithExpectedName(self.config)
        self.assertTrue(model.enable_shap)

        self.config.data["step_param_set"]["steps"]["model"]["calculate_shap"] = False
        model = ModelBaseWithExpectedName(self.config)
        self.assertFalse(model.enable_shap)

    def test_update_contingency_table_flow(self):
        """
        Ensure that the contingency table is correctly initialized and
        appended to when calling update_contingency_table multiple times.
        """
        model = ModelBaseWithExpectedName(self.config)

        # --- Batch 1 (e.g., Fold k=0) ---
        model.k = 0
        model.test_set = pl.DataFrame({"label": [0, 1, 0]})
        model.predictions = pl.DataFrame(
            {"label": [0, 1, 0], "predicted_label": [0, 1, 0], "score": [0.1, 0.9, 0.4]}
        )

        model.update_contingency_table()

        # Check initialization
        self.assertIsNotNone(model.contingency_table)
        self.assertEqual(model.contingency_table.shape, (3, 4))
        self.assertListEqual(
            model.contingency_table.columns, ["k", "label", "predicted_label", "score"]
        )

        # Verify content of Batch 1
        expected_batch_1 = pl.DataFrame(
            {
                "k": [0, 0, 0],
                "label": [0, 1, 0],
                "predicted_label": [0, 1, 0],
                "score": [0.1, 0.9, 0.4],
            }
        )
        self.assertTrue(model.contingency_table.equals(expected_batch_1))

        # --- Batch 2 (e.g., Fold k=1) ---
        model.k = 1
        model.test_set = pl.DataFrame({"label": [1, 1]})
        model.predictions = pl.DataFrame(
            {"label": [1, 0], "predicted_label": [1, 0], "score": [0.8, 0.3]}
        )

        model.update_contingency_table()

        # Check appending behavior
        self.assertEqual(model.contingency_table.shape, (5, 4))

        # Verify that k=1 rows were added
        k1_rows = model.contingency_table.filter(pl.col("k") == 1)
        self.assertEqual(k1_rows.shape, (2, 4))
        self.assertEqual(k1_rows["score"].to_list(), [0.8, 0.3])
