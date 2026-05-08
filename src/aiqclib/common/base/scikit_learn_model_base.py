"""
This module defines `SklearnModelBase`, an abstract base class for models
that adhere to the Scikit-Learn API (including XGBoost and native sklearn models).

It implements common workflows for data conversion, model building,
prediction, reporting, and SHAP value calculation for Explainable AI (XAI).
"""

import warnings
from typing import Any, Self, Optional

import polars as pl
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer

from aiqclib.common.base.config_base import ConfigBase
from aiqclib.common.base.model_base import ModelBase


class SklearnModelBase(ModelBase):
    """
    Abstract base class for Scikit-Learn compatible models.

    This class implements the standard lifecycle methods (:meth:`build`,
    :meth:`test`, :meth:`predict`, :meth:`create_report`) assuming the
    underlying model object supports the standard ``fit``, ``predict``,
    and ``predict_proba`` methods.

    It also integrates SHAP (SHapley Additive exPlanations) to provide feature
    importance values. SHAP calculation is controlled by the `calculate_shap`
    configuration flag, and can be overridden via `self.enable_shap` to
    disable it during computationally heavy steps like k-fold validation.

    Subclasses must implement:
      - :meth:`_get_model_class`: To return the specific class type.
    """

    def __init__(self, config: ConfigBase) -> None:
        """
        Initialize the model base.

        :param config: A configuration object containing model and step parameters.
        :type config: ConfigBase
        """
        super().__init__(config=config)

        # Check config to see if SHAP should be calculated
        self.enable_shap: bool = self.config.get_step_params("model").get(
            "calculate_shap", False
        )

        # Initialize storage for SHAP values explicitly
        self.shap_values: Optional[pl.DataFrame] = None

    def _get_model_class(self) -> Any:
        """
        Placeholder method.

        :return: The class object (e.g., xgboost.XGBClassifier, sklearn.linear_model.LogisticRegression).
        :rtype: Any
        """
        pass

    def build(self) -> None:
        """
        Train the classifier using the assigned training set.

        Steps:
          1. Convert the Polars DataFrame (:attr:`training_set`) to Pandas.
          2. Separate features (X) and labels (y).
          3. Initialize the model class provided by :meth:`_get_model_class`
             with :attr:`model_params`.
          4. Fit the model.

        :raises ValueError: If :attr:`training_set` is ``None`` or empty.
        """
        if self.training_set is None:
            raise ValueError("Member variable 'training_set' must not be empty.")

        x_train = self.training_set.select(pl.exclude("label")).to_pandas()
        if not self.allow_na:
            imputer = SimpleImputer(strategy="median")
            x_train = pd.DataFrame(
                imputer.fit_transform(x_train),
                columns=x_train.columns,  # keep column names
            )
        y_train = self.training_set["label"].to_pandas()

        model_class = self._get_model_class()
        self.model = model_class(**self.model_params)
        self.model.fit(x_train, y_train)

    def test(self) -> None:
        """
        Evaluate the trained classifier on the assigned test set.

        Steps:
          1. Call :meth:`predict` to generate predictions on the test set.
          2. Call :meth:`create_report` to compute metrics.
          3. Call :meth:`update_contingency_table` to store scores.
          4. Call :meth:`calculate_shap` to compute feature importances (if enabled).

        :raises ValueError: If :attr:`test_set` is ``None``.
        """
        self.predict()
        self.create_report()
        self.update_contingency_table()

        if self.enable_shap:
            self.calculate_shap()

    def update_nthreads(self, model: Self) -> Self:
        """
        Update the number of threads set in the model.

        :param model: The model instance whose thread count needs to be updated.
        :type model: Self
        :return: The updated model instance.
        :rtype: Self
        """
        if "n_jobs" in self.model_params and hasattr(model.model, "n_jobs"):
            model.model.n_jobs = self.model_params["n_jobs"]

        return model

    def predict(self) -> None:
        """
        Generates predictions for the test set using the trained model.

        Converts the Polars test set to a Pandas DataFrame, makes predictions,
        and stores the results in :attr:`predictions`.

        :raises ValueError: If :attr:`test_set` is ``None``.
        """
        if self.test_set is None:
            raise ValueError("Member variable 'test_set' must not be empty.")

        x_test = self.test_set.select(pl.exclude("label")).to_pandas()

        if self.allow_na:
            self.predictions = pl.DataFrame(
                {
                    "predicted_label": self.model.predict(x_test),
                    "score": self.model.predict_proba(x_test)[:, 1],
                }
            )
        else:
            self.safe_predict()

    def safe_predict(self) -> None:
        x_test = self.test_set.select(pl.exclude("label")).to_pandas()

        nan_rows = x_test.isna().any(axis=1).to_numpy()

        predictions = np.zeros(len(x_test), dtype=int)
        probs = np.zeros((len(x_test), 2))
        probs[:, 0] = 1.0

        if (~nan_rows).any():
            predictions[~nan_rows] = self.model.predict(x_test[~nan_rows])
            probs[~nan_rows] = self.model.predict_proba(x_test[~nan_rows])

        self.predictions = pl.DataFrame(
            {
                "predicted_label": predictions,
                "score": probs[:, 1],
            }
        )

    def calculate_shap(self) -> None:
        """
        Calculates SHAP values for the test set based on the specific model type.

        It automatically selects the optimal Explainer (`TreeExplainer`, `LinearExplainer`,
        or `KernelExplainer`). SHAP results are formatted into a Polars DataFrame
        and stored in :attr:`shap_values`.

        :raises ValueError: If :attr:`test_set` or :attr:`predictions` are ``None``.
        """
        if self.test_set is None:
            raise ValueError(
                "Member variable 'test_set' must not be empty to calculate SHAP."
            )

        if self.predictions is None:
            raise ValueError("Member variable 'predictions' must not be empty.")

        # Import shap inline to avoid heavy dependency loading if SHAP is disabled
        import shap
        import numpy as np

        x_test = self.test_set.select(pl.exclude("label")).to_pandas()

        # Determine optimal background data for explainers that require it
        if self.training_set is not None:
            background_data = self.training_set.select(pl.exclude("label")).to_pandas()
        else:
            # Fallback to test data if training data is unavailable (e.g., classification phase)
            background_data = x_test

        model_name = getattr(self, "expected_class_name", "Unknown")

        # 1. Tree Models (Fast & Exact)
        if model_name in ["XGBoost", "RandomForest", "DecisionTree"]:
            explainer = shap.TreeExplainer(self.model)
            shap_output = explainer.shap_values(x_test)

        # 2. Linear Models (Fast)
        elif model_name in ["LogisticRegression", "LinearDiscriminantAnalysis"]:
            explainer = shap.LinearExplainer(self.model, background_data)
            shap_output = explainer.shap_values(x_test)

        # 3. Model-Agnostic / Neural Models (Slow)
        else:
            warnings.warn(
                f"Using slow KernelExplainer for {model_name}. This may take a while."
            )
            # Summarize background data heavily to prevent massive slowdowns
            background_summary = shap.kmeans(
                background_data, min(100, background_data.shape[0])
            )

            explainer = shap.KernelExplainer(
                self.model.predict_proba, background_summary
            )
            shap_output = explainer.shap_values(x_test)

        # --- STANDARDIZE SHAP OUTPUT SHAPE ---
        # 1. Handle lists (RandomForest/DecisionTree returns [array_class0, array_class1])
        if isinstance(shap_output, list):
            # Take the positive class (index 1) if binary classification
            shap_output = shap_output[1] if len(shap_output) > 1 else shap_output[0]

        # 2. Handle 3D arrays (Some explainers return (n_samples, n_features, n_classes))
        if len(shap_output.shape) == 3:
            # Take the positive class (index 1)
            shap_output = shap_output[:, :, 1]

        # Create dictionary explicitly converting to 1D float64 arrays
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

    def create_report(self) -> None:
        """
        Computes and compiles a comprehensive classification report based on test results.

        Calculates precision, recall, f1-score, and support using
        :func:`sklearn.metrics.classification_report`. Stores the result
        in :attr:`report`.

        :raises ValueError: If :attr:`test_set` or :attr:`predictions` are ``None``.
        """
        if self.test_set is None:
            raise ValueError("Member variable 'test_set' must not be empty.")

        if self.predictions is None:
            raise ValueError("Member variable 'predictions' must not be empty.")

        y_test = self.test_set["label"].to_pandas()
        y_pred = self.predictions["predicted_label"].to_pandas()

        classification_dict = classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        )

        report_rows = []

        for label_key, metrics in classification_dict.items():
            if label_key == "accuracy":
                report_rows.append(
                    {"k": self.k, "metric_type": "overall_accuracy", "value": metrics}
                )
            elif label_key == "macro avg":
                balanced_accuracy = metrics.get("recall")
                report_rows.append(
                    {
                        "k": self.k,
                        "metric_type": "balanced_accuracy",
                        "value": balanced_accuracy,
                    }
                )
                report_rows.append(
                    {
                        "k": self.k,
                        "metric_type": "classification_report",
                        "label": label_key,
                        "precision": metrics.get("precision"),
                        "recall": metrics.get("recall"),
                        "f1-score": metrics.get("f1-score"),
                        "support": metrics.get("support"),
                    }
                )
            else:
                report_rows.append(
                    {
                        "k": self.k,
                        "metric_type": "classification_report",
                        "label": label_key,
                        "precision": metrics.get("precision"),
                        "recall": metrics.get("recall"),
                        "f1-score": metrics.get("f1-score"),
                        "support": metrics.get("support"),
                    }
                )

        self.report = pl.DataFrame(report_rows)

        if self.k == 0:
            self.report = self.report.drop("k")
