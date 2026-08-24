"""
This module defines `SklearnRegressorModelBase`, the base class for
regressor models used with proportion labels (floats in [0, 1], e.g. the
fraction of bad-flagged observations in a profile).

It reuses the Scikit-Learn lifecycle of
:class:`~aiqclib.common.base.scikit_learn_model_base.SklearnModelBase` but
predicts with ``model.predict`` (clipped to [0, 1]) instead of
``predict_proba``, reports regression metrics instead of a classification
report, and computes SHAP values without a class axis. The prediction frame
keeps the ``predicted_label`` / ``score`` schema: ``score`` is the clipped
prediction and ``predicted_label`` thresholds it, meaning "flag when the
predicted bad fraction reaches ``predicted_label_threshold``".
"""

import warnings

import numpy as np
import polars as pl
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from aiqclib.common.base.scikit_learn_model_base import SklearnModelBase
from aiqclib.common.utils.diagnostics import report_shap_cost, warn_constant_labels


class SklearnRegressorModelBase(SklearnModelBase):
    """
    Abstract base class for Scikit-Learn compatible regressor models.

    Subclasses must implement :meth:`_get_model_class` returning a regressor
    class (e.g. ``xgboost.XGBRegressor``).
    """

    is_regressor = True

    def _check_training_labels(self, labels: pl.Series) -> None:
        """
        Warn (rather than refuse) when the training labels are constant.

        A constant continuous label yields a constant predictor: degenerate
        but possible on legitimately clean data, unlike the single-class
        classifier case.

        :param labels: The training labels.
        :type labels: pl.Series
        """
        warn_constant_labels(labels, self.target_name, self.k or 0)

    def predict(self) -> None:
        """
        Generate predictions for the test set using the trained regressor.

        ``score`` is ``model.predict`` clipped to [0, 1] (the label domain);
        ``predicted_label`` thresholds the score with
        :attr:`predicted_label_threshold`, keeping the prediction schema of
        the classifier pipeline.

        :raises ValueError: If :attr:`test_set` is ``None``.
        """
        if self.test_set is None:
            raise ValueError("Member variable 'test_set' must not be empty.")

        x_test = self.test_set.select(pl.exclude("label")).to_pandas()

        if self.allow_na:
            scores = np.clip(self.model.predict(x_test), 0.0, 1.0)
            self.predictions = pl.DataFrame(
                {
                    "predicted_label": (
                        scores >= self.predicted_label_threshold
                    ).astype(int),
                    "score": scores,
                }
            )
        else:
            self.safe_predict()

    def safe_predict(self) -> None:
        """
        Predict with NaN-carrying rows pinned to a score of 0.0.

        Mirrors the classifier's ``safe_predict``: rows with missing feature
        values keep score 0.0 (below any positive threshold, so label 0).
        """
        x_test = self.test_set.select(pl.exclude("label")).to_pandas()

        nan_rows = x_test.isna().any(axis=1).to_numpy()

        scores = np.zeros(len(x_test))
        if (~nan_rows).any():
            scores[~nan_rows] = np.clip(self.model.predict(x_test[~nan_rows]), 0.0, 1.0)

        self.predictions = pl.DataFrame(
            {
                "predicted_label": (scores >= self.predicted_label_threshold).astype(
                    int
                ),
                "score": scores,
            }
        )

    def create_report(self) -> None:
        """
        Compile a regression report over the test predictions.

        Produces one row per metric (``mae``, ``rmse``, ``r2`` and
        ``n_samples``) in the same long-format frame the report writers
        serialize for classification reports.

        :raises ValueError: If :attr:`test_set` or :attr:`predictions` are ``None``.
        """
        if self.test_set is None:
            raise ValueError("Member variable 'test_set' must not be empty.")

        if self.predictions is None:
            raise ValueError("Member variable 'predictions' must not be empty.")

        y_test = self.test_set["label"].to_numpy()
        y_pred = self.predictions["score"].to_numpy()

        report_rows = [
            {
                "k": self.k,
                "metric_type": "mae",
                "value": float(mean_absolute_error(y_test, y_pred)),
            },
            {
                "k": self.k,
                "metric_type": "rmse",
                "value": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            },
            {
                "k": self.k,
                "metric_type": "r2",
                "value": float(r2_score(y_test, y_pred))
                if len(np.unique(y_test)) > 1
                else None,
            },
            {
                "k": self.k,
                "metric_type": "n_samples",
                "value": float(len(y_test)),
            },
        ]

        self.report = pl.DataFrame(report_rows)

        if self.k == 0:
            self.report = self.report.drop("k")

    def calculate_shap(self) -> None:
        """
        Calculate SHAP values for the test set for a regressor model.

        Tree-based regressors use the fast ``TreeExplainer``; anything else
        falls back to the model-agnostic ``KernelExplainer`` over
        ``model.predict``. Regressor SHAP output has no class axis, so no
        positive-class selection is needed.

        :raises ValueError: If :attr:`test_set` or :attr:`predictions` are ``None``.
        """
        if self.test_set is None:
            raise ValueError(
                "Member variable 'test_set' must not be empty to calculate SHAP."
            )

        if self.predictions is None:
            raise ValueError("Member variable 'predictions' must not be empty.")

        import shap

        x_test = self.test_set.select(pl.exclude("label")).to_pandas()

        report_shap_cost(x_test.shape[0], target_name=self.target_name, k=self.k or 0)

        model_name = getattr(self, "expected_class_name", "Unknown")

        if model_name in ["XGBoostRegressor", "RandomForestRegressor"]:
            explainer = shap.TreeExplainer(self.model)
            shap_output = explainer.shap_values(x_test)
        else:
            warnings.warn(
                f"Using slow KernelExplainer for {model_name}. This may take a while."
            )
            if self.training_set is not None:
                background = self.training_set.select(pl.exclude("label")).to_pandas()
            else:
                background = x_test
            background_summary = shap.kmeans(background, min(100, background.shape[0]))
            explainer = shap.KernelExplainer(self.model.predict, background_summary)
            shap_output = explainer.shap_values(x_test)

        if isinstance(shap_output, list):
            shap_output = shap_output[0]
        if len(shap_output.shape) != 2:
            raise ValueError(
                f"Unexpected SHAP output shape {shap_output.shape} for "
                f"regressor '{model_name}'; expected (n_samples, n_features)."
            )

        feature_names = x_test.columns.tolist()
        shap_cols = {
            f"{col}_shap": np.array(shap_output[:, i], dtype=np.float64).flatten()
            for i, col in enumerate(feature_names)
        }

        current_data = pl.DataFrame(
            {
                "label": self.test_set["label"],
                "predicted_label": self.predictions["predicted_label"],
                "score": self.predictions["score"],
            }
        )

        self.shap_values = pl.concat(
            [current_data, pl.DataFrame(shap_cols)], how="horizontal"
        )
