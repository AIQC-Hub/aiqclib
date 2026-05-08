"""
Unit tests for the SklearnModelBase class in aiqclib.common.base.scikit_learn_model_base.
This module verifies the correct functionality of the common Scikit-Learn API wrapper methods,
including the newly integrated SHAP (Explainable AI) functionalities.
"""

import unittest
import warnings
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import polars as pl
from sklearn.base import BaseEstimator, ClassifierMixin

from aiqclib.common.base.config_base import ConfigBase
from aiqclib.common.base.scikit_learn_model_base import SklearnModelBase
from aiqclib.common.config.training_config import TrainingConfig


class MockSklearnClassifier(BaseEstimator, ClassifierMixin):
    """
    A simple mock classifier compatible with Scikit-Learn API for testing purposes.
    """

    def __init__(self, **kwargs):
        self.params = kwargs
        self.n_jobs = kwargs.get("n_jobs", 1)

    def fit(self, X, y):
        return self

    def predict(self, X):
        # Return dummy predictions (all 0s)
        return np.zeros(X.shape[0])

    def predict_proba(self, X):
        # Return dummy probabilities (all 0.5s for class 1)
        # Shape: (n_samples, 2)
        n = X.shape[0]
        return np.column_stack((np.full(n, 0.5), np.full(n, 0.5)))


class ConcreteSklearnModel(SklearnModelBase):
    """
    Concrete implementation of SklearnModelBase for testing.
    """

    expected_class_name: str = (
        "XGBoost"  # Reusing a valid class name from config for simplicity
    )

    def __init__(self, config: ConfigBase) -> None:
        super().__init__(config)
        self.model_params = {"n_jobs": 1}

    def _get_model_class(self) -> Any:
        return MockSklearnClassifier


def make_training_set():
    return pl.DataFrame(
        {
            "f1": [1.0, 2.0, None, 4.0],
            "f2": [1.0, None, 3.0, 4.0],
            "label": [0, 1, 0, 1],
        }
    )


def make_test_set():
    return pl.DataFrame(
        {
            "f1": [1.0, None, 3.0],
            "f2": [1.0, 2.0, None],
            "label": [0, 0, 1],
        }
    )


class TestSklearnModelBase(unittest.TestCase):
    """
    A suite of tests that verify the correctness of methods within SklearnModelBase.
    """

    def setUp(self):
        """
        Set up configuration and a concrete model instance.
        """
        self.config_file_path = (
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_training_001.yaml"
        )
        self.config = TrainingConfig(str(self.config_file_path))
        self.config.select("NRT_BO_001")

        # Explicitly clear SHAP config for standard tests
        self.config.data["step_param_set"]["steps"]["model"]["calculate_shap"] = False
        self.model_wrapper = ConcreteSklearnModel(self.config)

    def test_init_shap_config(self):
        """
        Ensure SHAP settings are correctly parsed from the config during initialization.
        """
        # Default should be False
        self.assertFalse(self.model_wrapper.enable_shap)
        self.assertIsNone(self.model_wrapper.shap_values)

        # Override config to True
        self.config.data["step_param_set"]["steps"]["model"]["calculate_shap"] = True
        model_wrapper_shap = ConcreteSklearnModel(self.config)
        self.assertTrue(model_wrapper_shap.enable_shap)

    def test_build(self):
        """
        Ensure build converts data and fits the underlying model.
        """
        # Setup dummy training data
        self.model_wrapper.training_set = pl.DataFrame(
            {"feature1": [1.0, 2.0, 3.0], "label": [0, 1, 0]}
        )

        self.model_wrapper.build()

        self.assertIsInstance(self.model_wrapper.model, MockSklearnClassifier)

    def test_build_empty_training_set(self):
        """
        Ensure build raises ValueError if training_set is missing.
        """
        self.model_wrapper.training_set = None
        with self.assertRaisesRegex(ValueError, "training_set"):
            self.model_wrapper.build()

    def test_predict(self):
        """
        Ensure predict generates predictions and scores in the correct format.
        """
        # We need a fitted model (or just an instance for the mock)
        self.model_wrapper.model = MockSklearnClassifier()
        self.model_wrapper.test_set = pl.DataFrame(
            {"feature1": [1.0, 2.0], "label": [0, 1]}
        )

        self.model_wrapper.predict()

        self.assertIsNotNone(self.model_wrapper.predictions)
        self.assertEqual(self.model_wrapper.predictions.shape, (2, 2))
        self.assertListEqual(
            self.model_wrapper.predictions.columns, ["predicted_label", "score"]
        )
        # Based on MockSklearnClassifier logic:
        self.assertEqual(self.model_wrapper.predictions["predicted_label"][0], 0.0)
        self.assertEqual(self.model_wrapper.predictions["score"][0], 0.5)

    def test_predict_empty_test_set(self):
        """
        Ensure predict raises ValueError if test_set is missing.
        """
        with self.assertRaisesRegex(ValueError, "test_set"):
            self.model_wrapper.predict()

    def test_create_report(self):
        """
        Ensure create_report generates a DataFrame with metrics.
        """
        self.model_wrapper.k = 1
        self.model_wrapper.test_set = pl.DataFrame({"label": [0, 1, 0, 1]})
        self.model_wrapper.predictions = pl.DataFrame(
            {
                "predicted_label": [0, 1, 0, 0],  # One error
                "score": [0.5, 0.5, 0.5, 0.5],
            }
        )

        self.model_wrapper.create_report()

        self.assertIsNotNone(self.model_wrapper.report)
        self.assertIsInstance(self.model_wrapper.report, pl.DataFrame)
        self.assertIn("k", self.model_wrapper.report.columns)
        self.assertIn("metric_type", self.model_wrapper.report.columns)
        self.assertIn(
            "value", self.model_wrapper.report.columns
        )  # For accuracy/balanced

        # Verify specific rows exist
        metrics = self.model_wrapper.report["metric_type"].unique().to_list()
        self.assertIn("overall_accuracy", metrics)
        self.assertIn("balanced_accuracy", metrics)
        self.assertIn("classification_report", metrics)

    def test_test_workflow_shap_disabled(self):
        """
        Ensure the test method calls base methods but skips SHAP if disabled.
        """
        self.model_wrapper.enable_shap = False

        self.model_wrapper.predict = MagicMock()
        self.model_wrapper.create_report = MagicMock()
        self.model_wrapper.update_contingency_table = MagicMock()
        self.model_wrapper.calculate_shap = MagicMock()

        self.model_wrapper.test()

        self.model_wrapper.predict.assert_called_once()
        self.model_wrapper.create_report.assert_called_once()
        self.model_wrapper.update_contingency_table.assert_called_once()
        self.model_wrapper.calculate_shap.assert_not_called()

    def test_test_workflow_shap_enabled(self):
        """
        Ensure the test method calls calculate_shap if enable_shap is True.
        """
        self.model_wrapper.enable_shap = True

        self.model_wrapper.predict = MagicMock()
        self.model_wrapper.create_report = MagicMock()
        self.model_wrapper.update_contingency_table = MagicMock()
        self.model_wrapper.calculate_shap = MagicMock()

        self.model_wrapper.test()

        self.model_wrapper.calculate_shap.assert_called_once()

    def test_update_nthreads(self):
        """
        Ensure update_nthreads updates the underlying model's n_jobs.
        """
        self.model_wrapper.model = MockSklearnClassifier(n_jobs=1)
        self.model_wrapper.model_params = {"n_jobs": 4}

        self.model_wrapper.update_nthreads(self.model_wrapper)

        self.assertEqual(self.model_wrapper.model.n_jobs, 4)

    def test_calculate_shap_missing_test_set(self):
        """
        Ensure calculate_shap raises ValueError if test_set is missing.
        """
        self.model_wrapper.test_set = None
        with self.assertRaisesRegex(ValueError, "test_set"):
            self.model_wrapper.calculate_shap()

    def test_calculate_shap_tree_explainer(self):
        """
        Ensure TreeExplainer is correctly routed and utilized for Tree models.
        Mocking the SHAP module prevents slow computations and dependency issues during tests.
        """
        with patch.dict("sys.modules", {"shap": MagicMock()}) as mock_sys_modules:
            mock_shap = mock_sys_modules["shap"]

            # Setup mock explainer returning standard array output (like XGBoost)
            mock_explainer = MagicMock()
            mock_explainer.shap_values.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
            mock_shap.TreeExplainer.return_value = mock_explainer

            self.model_wrapper.expected_class_name = "XGBoost"
            self.model_wrapper.model = MockSklearnClassifier()
            self.model_wrapper.test_set = pl.DataFrame(
                {"f1": [1.0, 2.0], "f2": [3.0, 4.0], "label": [0, 1]}
            )
            self.model_wrapper.predictions = pl.DataFrame(
                {"label": [0, 1], "predicted_label": [0, 1], "score": [0.1, 0.9]}
            )

            self.model_wrapper.calculate_shap()

            # Verify TreeExplainer was chosen
            mock_shap.TreeExplainer.assert_called_once_with(self.model_wrapper.model)

            # Verify Polars DataFrame was constructed properly
            self.assertIsNotNone(self.model_wrapper.shap_values)
            self.assertListEqual(
                self.model_wrapper.shap_values.columns,
                ["label", "predicted_label", "score", "f1_shap", "f2_shap"],
            )
            self.assertEqual(self.model_wrapper.shap_values["f1_shap"][0], 0.1)

    def test_calculate_shap_linear_explainer(self):
        """
        Ensure LinearExplainer is correctly routed for Linear models.
        """
        with patch.dict("sys.modules", {"shap": MagicMock()}) as mock_sys_modules:
            mock_shap = mock_sys_modules["shap"]

            mock_explainer = MagicMock()
            mock_explainer.shap_values.return_value = np.array([[0.5, 0.6]])
            mock_shap.LinearExplainer.return_value = mock_explainer

            self.model_wrapper.expected_class_name = "LogisticRegression"
            self.model_wrapper.model = MockSklearnClassifier()
            self.model_wrapper.training_set = pl.DataFrame(
                {"f1": [1.0], "f2": [2.0], "label": [0]}
            )
            self.model_wrapper.test_set = pl.DataFrame(
                {"f1": [1.0], "f2": [2.0], "label": [0]}
            )
            self.model_wrapper.predictions = pl.DataFrame(
                {"label": [0], "predicted_label": [0], "score": [0.4]}
            )

            self.model_wrapper.calculate_shap()

            # Verify LinearExplainer was chosen
            mock_shap.LinearExplainer.assert_called_once()

            self.assertIsNotNone(self.model_wrapper.shap_values)
            self.assertEqual(self.model_wrapper.shap_values["f1_shap"][0], 0.5)

    def test_calculate_shap_kernel_explainer(self):
        """
        Ensure KernelExplainer is routed with background summarization for Blackbox models.
        """
        with patch.dict("sys.modules", {"shap": MagicMock()}) as mock_sys_modules:
            mock_shap = mock_sys_modules["shap"]

            mock_explainer = MagicMock()
            # Kernel Explainer usually returns a list of arrays (one for each class)
            mock_explainer.shap_values.return_value = [
                np.array([[-0.1, -0.2]]),  # class 0
                np.array([[0.9, 0.8]]),  # class 1 (we extract this)
            ]
            mock_shap.KernelExplainer.return_value = mock_explainer
            mock_shap.kmeans.return_value = "mock_kmeans_summary"

            self.model_wrapper.expected_class_name = "SVM"
            self.model_wrapper.model = MockSklearnClassifier()
            self.model_wrapper.training_set = pl.DataFrame(
                {"f1": [1.0], "f2": [2.0], "label": [0]}
            )
            self.model_wrapper.test_set = pl.DataFrame(
                {"f1": [1.0], "f2": [2.0], "label": [0]}
            )
            self.model_wrapper.predictions = pl.DataFrame(
                {"label": [0], "predicted_label": [0], "score": [0.1]}
            )

            # Suppress the UserWarning triggered when routing to KernelExplainer
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model_wrapper.calculate_shap()

            # Verify K-means summarization was called
            mock_shap.kmeans.assert_called_once()
            # Verify KernelExplainer was initialized with predict_proba
            mock_shap.KernelExplainer.assert_called_once_with(
                self.model_wrapper.model.predict_proba, "mock_kmeans_summary"
            )

            # Verify correct class array (index 1) was extracted
            self.assertIsNotNone(self.model_wrapper.shap_values)
            self.assertEqual(self.model_wrapper.shap_values["f1_shap"][0], 0.9)

    def test_safe_predict_nan_rows_get_default_class(self):
        # Setup dummy training data
        self.model_wrapper.training_set = make_training_set()
        self.model_wrapper.test_set = make_test_set()
        self.model_wrapper.allow_na = False

        self.model_wrapper.build()
        self.model_wrapper.predict()

        preds = self.model_wrapper.predictions

        x_test = self.model_wrapper.test_set.select(pl.exclude("label")).to_pandas()
        nan_rows = pd.isna(x_test).any(axis=1)

        predicted = preds["predicted_label"].to_numpy()

        assert (predicted[nan_rows] == 0).all()

    def test_safe_predict_probability_range(self):
        # Setup dummy training data
        self.model_wrapper.training_set = make_training_set()
        self.model_wrapper.test_set = make_test_set()
        self.model_wrapper.allow_na = False

        self.model_wrapper.build()
        self.model_wrapper.predict()

        scores = self.model_wrapper.predictions["score"].to_numpy()

        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)

    def test_non_nan_rows_use_model_prediction(self):
        # Setup dummy training data
        self.model_wrapper.training_set = make_training_set()
        self.model_wrapper.test_set = make_test_set()
        self.model_wrapper.allow_na = False

        self.model_wrapper.build()
        self.model_wrapper.predict()

        x_test = self.model_wrapper.test_set.select(pl.exclude("label")).to_pandas()
        nan_rows = pd.isna(x_test).any(axis=1)

        non_nan_rows = ~nan_rows

        if non_nan_rows.any():
            expected = self.model_wrapper.model.predict(
                np.asarray(x_test)[non_nan_rows]
            )
            predicted = self.model_wrapper.predictions["predicted_label"].to_numpy()[
                non_nan_rows
            ]

            assert np.array_equal(predicted, expected)
