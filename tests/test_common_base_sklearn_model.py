"""Unit tests for the ``SklearnModelBase`` class.

Coverage spans the full life-cycle of a sklearn-wrapper model:
- Initialization: SHAP flag propagation
- ``build``: trains the underlying classifier; empty training set raises ValueError
- ``predict``: produces a (n_test, 2) frame of (predicted_label, score);
  empty test set raises ValueError
- ``create_report``: computes overall_accuracy, balanced_accuracy, and
  classification_report metrics
- ``test``: orchestrates predict → create_report → update_contingency_table;
  calls calculate_shap only if enable_shap is True (verified via MagicMock)
- ``update_nthreads``: pushes ``model_params["n_jobs"]`` onto the underlying model
- ``calculate_shap``: routes to TreeExplainer / LinearExplainer / KernelExplainer
  based on ``expected_class_name``; missing test_set raises ValueError
- Safe-predict edge cases: NaN rows get a default class; non-NaN rows
  use the model's actual prediction; scores stay in [0, 1]

Refactored from a ``unittest.TestCase`` class that was already heavily
mock-based (MagicMock + patch). Module-level mocks (MockSklearnClassifier,
ConcreteSklearnModel, make_training_set, make_test_set) stay at module
level. The setUp's "set calculate_shap=False then build wrapper" pattern
becomes a ``model_wrapper`` fixture.
"""

import warnings
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

from aiqclib.common.base.config_base import ConfigBase
from aiqclib.common.base.scikit_learn_model_base import SklearnModelBase


# ---------------------------------------------------------------------------
# Module-level mocks and helpers
# ---------------------------------------------------------------------------


class MockSklearnClassifier(BaseEstimator, ClassifierMixin):
    """A minimal sklearn-compatible classifier with deterministic dummy output.

    - ``fit``: no-op
    - ``predict``: returns all zeros
    - ``predict_proba``: returns 0.5 for both classes (shape (n, 2))
    """

    def __init__(self, **kwargs):
        self.params = kwargs
        self.n_jobs = kwargs.get("n_jobs", 1)

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(X.shape[0])

    def predict_proba(self, X):
        n = X.shape[0]
        return np.column_stack((np.full(n, 0.5), np.full(n, 0.5)))


class ConcreteSklearnModel(SklearnModelBase):
    """Concrete subclass exposing SklearnModelBase methods for testing.

    Reuses ``expected_class_name = "XGBoost"`` so the config-driven
    constructor validates against a real registered model class.
    """

    expected_class_name: str = "XGBoost"

    def __init__(self, config: ConfigBase) -> None:
        super().__init__(config)
        self.model_params = {"n_jobs": 1}

    def _get_model_class(self) -> Any:
        return MockSklearnClassifier


def make_training_set() -> pl.DataFrame:
    """Training set with one NaN per feature column, 4 rows."""
    return pl.DataFrame(
        {
            "f1": [1.0, 2.0, None, 4.0],
            "f2": [1.0, None, 3.0, 4.0],
            "label": [0, 1, 0, 1],
        }
    )


def make_test_set() -> pl.DataFrame:
    """Test set with one NaN per feature column, 3 rows."""
    return pl.DataFrame(
        {
            "f1": [1.0, None, 3.0],
            "f2": [1.0, 2.0, None],
            "label": [0, 0, 1],
        }
    )


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def model_wrapper(training_config_001):
    """Fresh ConcreteSklearnModel with calculate_shap explicitly set to False.

    The setUp in the original test set ``calculate_shap=False`` before
    constructing the wrapper, treating that as a starting invariant.
    Preserving that here so each test starts from the same state.
    """
    training_config_001.data["step_param_set"]["steps"]["model"]["calculate_shap"] = (
        False
    )
    return ConcreteSklearnModel(training_config_001)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSklearnModelBase:
    """Tests for SklearnModelBase's lifecycle and SHAP routing."""

    # ----- Initialization / SHAP config -----

    def test_init_shap_config(self, training_config_001, model_wrapper):
        """SHAP settings are read from the config at construction time.

        ``model_wrapper`` was built with ``calculate_shap=False``; verify
        that initial state, then re-construct with ``calculate_shap=True``
        and verify the override.
        """
        assert model_wrapper.enable_shap is False
        assert model_wrapper.shap_values is None

        # Override and reconstruct to pick up the new flag.
        training_config_001.data["step_param_set"]["steps"]["model"][
            "calculate_shap"
        ] = True
        model_wrapper_shap = ConcreteSklearnModel(training_config_001)
        assert model_wrapper_shap.enable_shap is True

    # ----- build -----

    def test_build(self, model_wrapper):
        """build() converts the training_set frame and fits the classifier."""
        model_wrapper.training_set = pl.DataFrame(
            {
                "feature1": [1.0, 2.0, 3.0],
                "label": [0, 1, 0],
            }
        )

        model_wrapper.build()

        assert isinstance(model_wrapper.model, MockSklearnClassifier)

    def test_build_empty_training_set(self, model_wrapper):
        """build() with training_set=None raises ValueError mentioning training_set."""
        model_wrapper.training_set = None
        with pytest.raises(ValueError, match="training_set"):
            model_wrapper.build()

    # ----- predict -----

    def test_predict(self, model_wrapper):
        """predict() produces a 2-column (predicted_label, score) frame."""
        model_wrapper.model = MockSklearnClassifier()
        model_wrapper.test_set = pl.DataFrame(
            {
                "feature1": [1.0, 2.0],
                "label": [0, 1],
            }
        )

        model_wrapper.predict()

        assert model_wrapper.predictions is not None
        assert model_wrapper.predictions.shape == (2, 2)
        assert model_wrapper.predictions.columns == ["predicted_label", "score"]
        # MockSklearnClassifier returns all 0s for predict, 0.5 for score.
        assert model_wrapper.predictions["predicted_label"][0] == 0.0
        assert model_wrapper.predictions["score"][0] == 0.5

    def test_predict_empty_test_set(self, model_wrapper):
        """predict() with test_set=None raises ValueError mentioning test_set."""
        with pytest.raises(ValueError, match="test_set"):
            model_wrapper.predict()

    # ----- create_report -----

    def test_create_report(self, model_wrapper):
        """create_report() produces a metrics DataFrame with k/metric_type/value columns."""
        model_wrapper.k = 1
        model_wrapper.test_set = pl.DataFrame({"label": [0, 1, 0, 1]})
        model_wrapper.predictions = pl.DataFrame(
            {
                "predicted_label": [0, 1, 0, 0],  # one error → not full accuracy
                "score": [0.5, 0.5, 0.5, 0.5],
            }
        )

        model_wrapper.create_report()

        assert model_wrapper.report is not None
        assert isinstance(model_wrapper.report, pl.DataFrame)
        for col in ("k", "metric_type", "value"):
            assert col in model_wrapper.report.columns

        # Expected metrics present in the metric_type column.
        metric_types = model_wrapper.report["metric_type"].unique().to_list()
        for expected in (
            "overall_accuracy",
            "balanced_accuracy",
            "classification_report",
        ):
            assert expected in metric_types

    # ----- test() workflow (MagicMock-based) -----

    def test_test_workflow_shap_disabled(self, model_wrapper):
        """test() calls predict / create_report / update_contingency_table,
        but skips calculate_shap when enable_shap is False."""
        model_wrapper.enable_shap = False

        model_wrapper.predict = MagicMock()
        model_wrapper.create_report = MagicMock()
        model_wrapper.update_contingency_table = MagicMock()
        model_wrapper.calculate_shap = MagicMock()

        model_wrapper.test()

        model_wrapper.predict.assert_called_once()
        model_wrapper.create_report.assert_called_once()
        model_wrapper.update_contingency_table.assert_called_once()
        model_wrapper.calculate_shap.assert_not_called()

    def test_test_workflow_shap_enabled(self, model_wrapper):
        """test() calls calculate_shap when enable_shap is True."""
        model_wrapper.enable_shap = True

        model_wrapper.predict = MagicMock()
        model_wrapper.create_report = MagicMock()
        model_wrapper.update_contingency_table = MagicMock()
        model_wrapper.calculate_shap = MagicMock()

        model_wrapper.test()

        model_wrapper.calculate_shap.assert_called_once()

    # ----- update_nthreads -----

    def test_update_nthreads(self, model_wrapper):
        """update_nthreads pushes model_params['n_jobs'] onto the underlying model."""
        model_wrapper.model = MockSklearnClassifier(n_jobs=1)
        model_wrapper.model_params = {"n_jobs": 4}

        model_wrapper.update_nthreads(model_wrapper)

        assert model_wrapper.model.n_jobs == 4

    # ----- calculate_shap -----

    def test_calculate_shap_missing_test_set(self, model_wrapper):
        """calculate_shap() with test_set=None raises ValueError mentioning test_set."""
        model_wrapper.test_set = None
        with pytest.raises(ValueError, match="test_set"):
            model_wrapper.calculate_shap()

    def test_calculate_shap_tree_explainer(self, model_wrapper):
        """TreeExplainer is selected for tree-based models (XGBoost, RF, DT).

        Patches sys.modules['shap'] with a MagicMock to avoid the real SHAP
        library's slow computation and dependency requirements.
        """
        with patch.dict("sys.modules", {"shap": MagicMock()}) as mock_sys_modules:
            mock_shap = mock_sys_modules["shap"]

            # Mock explainer returning an array (tree-explainer's standard output shape).
            mock_explainer = MagicMock()
            mock_explainer.shap_values.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
            mock_shap.TreeExplainer.return_value = mock_explainer

            model_wrapper.expected_class_name = "XGBoost"
            model_wrapper.model = MockSklearnClassifier()
            model_wrapper.test_set = pl.DataFrame(
                {
                    "f1": [1.0, 2.0],
                    "f2": [3.0, 4.0],
                    "label": [0, 1],
                }
            )
            model_wrapper.predictions = pl.DataFrame(
                {
                    "label": [0, 1],
                    "predicted_label": [0, 1],
                    "score": [0.1, 0.9],
                }
            )

            model_wrapper.calculate_shap()

            # TreeExplainer was the routed choice.
            mock_shap.TreeExplainer.assert_called_once_with(model_wrapper.model)

            # Output frame is a Polars DataFrame with the expected columns.
            assert model_wrapper.shap_values is not None
            assert model_wrapper.shap_values.columns == [
                "label",
                "predicted_label",
                "score",
                "f1_shap",
                "f2_shap",
            ]
            assert model_wrapper.shap_values["f1_shap"][0] == 0.1

    def test_calculate_shap_linear_explainer(self, model_wrapper):
        """LinearExplainer is selected for linear models (LogisticRegression, LDA)."""
        with patch.dict("sys.modules", {"shap": MagicMock()}) as mock_sys_modules:
            mock_shap = mock_sys_modules["shap"]

            mock_explainer = MagicMock()
            mock_explainer.shap_values.return_value = np.array([[0.5, 0.6]])
            mock_shap.LinearExplainer.return_value = mock_explainer

            model_wrapper.expected_class_name = "LogisticRegression"
            model_wrapper.model = MockSklearnClassifier()
            model_wrapper.training_set = pl.DataFrame(
                {"f1": [1.0], "f2": [2.0], "label": [0]},
            )
            model_wrapper.test_set = pl.DataFrame(
                {"f1": [1.0], "f2": [2.0], "label": [0]},
            )
            model_wrapper.predictions = pl.DataFrame(
                {"label": [0], "predicted_label": [0], "score": [0.4]},
            )

            model_wrapper.calculate_shap()

            mock_shap.LinearExplainer.assert_called_once()
            assert model_wrapper.shap_values is not None
            assert model_wrapper.shap_values["f1_shap"][0] == 0.5

    def test_calculate_shap_kernel_explainer(self, model_wrapper):
        """KernelExplainer is selected for blackbox models (SVM, KNN, etc).

        KernelExplainer is the slowest, used as fallback. It also requires
        background data, which is summarized via ``shap.kmeans``. The mock
        verifies that pipeline.

        KernelExplainer returns one array per class; the wrapper picks
        index 1 (the positive class), so the extracted value should be 0.9
        rather than -0.1.
        """
        with patch.dict("sys.modules", {"shap": MagicMock()}) as mock_sys_modules:
            mock_shap = mock_sys_modules["shap"]

            mock_explainer = MagicMock()
            # KernelExplainer's per-class output: list of arrays.
            mock_explainer.shap_values.return_value = [
                np.array([[-0.1, -0.2]]),  # class 0
                np.array([[0.9, 0.8]]),  # class 1 (the wrapper extracts this)
            ]
            mock_shap.KernelExplainer.return_value = mock_explainer
            mock_shap.kmeans.return_value = "mock_kmeans_summary"

            model_wrapper.expected_class_name = "SVM"
            model_wrapper.model = MockSklearnClassifier()
            model_wrapper.training_set = pl.DataFrame(
                {"f1": [1.0], "f2": [2.0], "label": [0]},
            )
            model_wrapper.test_set = pl.DataFrame(
                {"f1": [1.0], "f2": [2.0], "label": [0]},
            )
            model_wrapper.predictions = pl.DataFrame(
                {"label": [0], "predicted_label": [0], "score": [0.1]},
            )

            # Suppress the UserWarning for routing to KernelExplainer.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model_wrapper.calculate_shap()

            # K-means summarization was invoked.
            mock_shap.kmeans.assert_called_once()
            # KernelExplainer initialized with predict_proba and the kmeans summary.
            mock_shap.KernelExplainer.assert_called_once_with(
                model_wrapper.model.predict_proba,
                "mock_kmeans_summary",
            )

            # Correct class array (index 1) extracted.
            assert model_wrapper.shap_values is not None
            assert model_wrapper.shap_values["f1_shap"][0] == 0.9

    # ----- Safe-predict edge cases -----

    def test_safe_predict_nan_rows_get_default_class(self, model_wrapper):
        """Rows with NaN features get the default class (0) instead of a model prediction."""
        model_wrapper.training_set = make_training_set()
        model_wrapper.test_set = make_test_set()
        model_wrapper.allow_na = False

        model_wrapper.build()
        model_wrapper.predict()

        preds = model_wrapper.predictions
        x_test = model_wrapper.test_set.select(pl.exclude("label")).to_pandas()
        nan_rows = pd.isna(x_test).any(axis=1)

        predicted = preds["predicted_label"].to_numpy()
        assert (predicted[nan_rows] == 0).all()

    def test_safe_predict_probability_range(self, model_wrapper):
        """All predicted scores stay in [0, 1]."""
        model_wrapper.training_set = make_training_set()
        model_wrapper.test_set = make_test_set()
        model_wrapper.allow_na = False

        model_wrapper.build()
        model_wrapper.predict()

        scores = model_wrapper.predictions["score"].to_numpy()
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)

    def test_non_nan_rows_use_model_prediction(self, model_wrapper):
        """Rows without NaN use the underlying model's actual prediction."""
        model_wrapper.training_set = make_training_set()
        model_wrapper.test_set = make_test_set()
        model_wrapper.allow_na = False

        model_wrapper.build()
        model_wrapper.predict()

        x_test = model_wrapper.test_set.select(pl.exclude("label")).to_pandas()
        nan_rows = pd.isna(x_test).any(axis=1)
        non_nan_rows = ~nan_rows

        if non_nan_rows.any():
            expected = model_wrapper.model.predict(
                np.asarray(x_test)[non_nan_rows],
            )
            predicted = model_wrapper.predictions["predicted_label"].to_numpy()[
                non_nan_rows
            ]
            assert np.array_equal(predicted, expected)
