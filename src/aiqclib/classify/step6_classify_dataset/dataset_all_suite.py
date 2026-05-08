"""
This module provides the ClassifyAllSuite class, which extends BuildModelBase to
facilitate the testing and evaluation of multiple classification models across
various targets and machine learning methods. It automates the process of
loading models, generating predictions, and aggregating results into unified
datasets for comparative analysis.
"""

import os
import copy
from typing import Optional, Dict

import polars as pl

from aiqclib.common.base.config_base import ConfigBase
from aiqclib.train.step4_build_model.build_model_base import BuildModelBase
from aiqclib.common.utils.metric_plots import create_multi_method_metric_plots
from aiqclib.common.loader.single_model_loader import (
    load_single_model_class_with_class_name,
)


class ClassifyAllSuite(BuildModelBase):
    """
    A subclass of :class:`BuildModelBase` that orchestrates the evaluation
    and testing of classification models for multiple targets using
    multiple machine learning methods provided by a ModelSuite.

    This class reads previously trained models (with composite keys) and
    aggregates test reports, predictions, and contingency tables into single
    datasets per target by introducing a 'method' column.

    .. note::
       This class sets :attr:`expected_class_name` to ``"ClassifyAllSuite"``.
    """

    expected_class_name: str = "ClassifyAllSuite"

    def __init__(
        self,
        config: ConfigBase,
        test_sets: Optional[Dict[str, pl.DataFrame]] = None,
    ) -> None:
        """
        Initialize the ClassifyAllSuite instance.

        :param config: A training configuration object specifying paths,
                       parameters, and model-building directives.
        :type config: ConfigBase
        :param test_sets: A dictionary of test data keyed by target name.
        :type test_sets: Optional[Dict[str, pl.DataFrame]]
        """
        super().__init__(
            config=config, training_sets=None, test_sets=test_sets, step_name="classify"
        )

        if not getattr(self.base_model, "multi", False):
            raise ValueError(
                "ClassifyAllSuite requires a base model with 'multi=True' "
                "(e.g., ModelSuite), but received a standard model class."
            )

        self.drop_cols = ["row_id", "platform_code", "profile_no", "observation_no"]
        self.test_cols = [
            "row_id",
            "platform_code",
            "profile_no",
            "observation_no",
            "label",
        ]

        # Consolidated files per target for data, but unique files per model/method
        self.default_file_names: Dict[str, str] = {
            "report": "classify_report_{target_name}.tsv",
            "prediction": "classify_prediction_{target_name}.parquet",
            "contingency_table": "classify_contingency_tables_{target_name}.parquet",
            "shap_value": "classify_shap_values_{target_name}.parquet",
            "metric_plot": "classify_metric_plots_{target_name}.svg",
        }
        self.default_model_file_name: str = "model_{method}_{target_name}.joblib"

        # Populate base path mappings (using standard template for aggregated data)
        self.output_file_names: Dict[str, Dict[str, str]] = {
            k: self.config.get_target_file_names("classify", v)
            for k, v in self.default_file_names.items()
        }

        # Populate paths specifically for individual model .joblib files
        base_models = self.config.get_target_file_names(
            step_name="model",
            default_file_name=self.default_model_file_name,
            use_dataset_folder=False,
        )

        self.model_file_names = {}
        for target_name in self.config.get_target_names():
            for method_name, method_obj in self.base_model.method_objs.items():
                method_lower = getattr(method_obj, "short_name", method_name).lower()
                comp_key = f"{method_lower}_{target_name}"
                self.model_file_names[comp_key] = base_models[target_name].replace(
                    "{method}", method_lower
                )

    def test_targets(self) -> None:
        """
        Iterate over all targets, ensuring that models have been read/loaded for all
        configured methods before calling :meth:`test`.

        :raises ValueError: If a target/method combination has no corresponding entry
                            in :attr:`models`.
        """
        for target_name in self.config.get_target_names():
            for method_name, method_obj in self.base_model.method_objs.items():
                method_lower = getattr(method_obj, "short_name", method_name).lower()
                comp_key = f"{method_lower}_{target_name}"

                if comp_key not in self.models:
                    raise ValueError(
                        f"No valid model found for the variable '{target_name}' "
                        f"and method '{method_name}' (expected key '{comp_key}')."
                    )
            self.test(target_name)

    def build(self, target_name: str) -> None:
        """
        Placeholder method as training does not occur during classification.

        :param target_name: The name of the target variable.
        :type target_name: str
        """
        pass  # pragma: no cover

    def build_final_model(self, target_name: str) -> None:
        """
        Placeholder method as training does not occur during classification.

        :param target_name: The name of the target variable.
        :type target_name: str
        """
        pass  # pragma: no cover

    def test(self, target_name: str) -> None:
        """
        Test the models for the given target across all methods, appending a
        'method' column and aggregating the results into single datasets.

        Data types for model outputs (class, score, etc.) are standardized
        to Int64 and Float64 to prevent Polars SchemaErrors when concatenating.

        :param target_name: The name of the target variable to be tested.
        :type target_name: str
        """
        test_set = self.test_sets[target_name].drop(self.drop_cols)

        target_reports = []
        target_predictions = []
        target_contingency = []
        target_shap_values = []

        for method_name, method_obj in self.base_model.method_objs.items():
            method_lower = getattr(method_obj, "short_name", method_name).lower()
            comp_key = f"{method_lower}_{target_name}"

            current_model = self.models[comp_key]
            current_model.contingency_table = None  # Reset to prevent duplication
            current_model.test_set = test_set
            current_model.test()

            # Append method column to report and normalize potential mixed int/float types
            if current_model.report is not None:
                rep_df = current_model.report.with_columns(
                    [pl.lit(method_name).alias("method")]
                )
                if "support" in rep_df.columns:
                    rep_df = rep_df.with_columns(pl.col("support").cast(pl.Float64))
                target_reports.append(rep_df.select(["method", pl.exclude("method")]))

            # Append method column to predictions and standardize prediction types
            pred_df = pl.concat(
                [
                    self.test_sets[target_name].select(self.test_cols),
                    current_model.predictions,
                ],
                how="horizontal",
            )
            pred_df = pred_df.with_columns(
                [
                    pl.lit(method_name).alias("method"),
                    pl.col("predicted_label").cast(pl.Int64),
                    pl.col("score").cast(pl.Float64),
                ]
            )
            target_predictions.append(pred_df.select(["method", pl.exclude("method")]))

            # Append method column to contingency table and standardize prediction types
            if current_model.contingency_table is not None:
                ct_df = current_model.contingency_table.with_columns(
                    [
                        pl.lit(method_name).alias("method"),
                        pl.col("k").cast(pl.Int64),
                        pl.col("predicted_label").cast(pl.Int64),
                        pl.col("score").cast(pl.Float64),
                    ]
                )
                target_contingency.append(
                    ct_df.select(["method", pl.exclude("method")])
                )

            # Append method column to shap values and standardize prediction types
            if current_model.shap_values is not None:
                shap_df = current_model.shap_values.with_columns(
                    [
                        pl.lit(method_name).alias("method"),
                        pl.col("predicted_label").cast(pl.Int64),
                        pl.col("score").cast(pl.Float64),
                    ]
                )

                # Explicitly cast all SHAP columns to Float64 just to be safe
                shap_features = [c for c in shap_df.columns if c.endswith("_shap")]
                if shap_features:
                    shap_df = shap_df.with_columns(
                        [pl.col(c).cast(pl.Float64) for c in shap_features]
                    )

                target_shap_values.append(
                    shap_df.select(["method", pl.exclude("method")])
                )

        self.reports[target_name] = (
            pl.concat(target_reports) if target_reports else None
        )
        self.predictions[target_name] = (
            pl.concat(target_predictions) if target_predictions else None
        )
        self.contingency_tables[target_name] = (
            pl.concat(target_contingency) if target_contingency else None
        )
        self.shap_values[target_name] = (
            pl.concat(target_shap_values) if target_shap_values else None
        )

    def read_models(self) -> None:
        """
        Read and restore each target's models from disk for all methods in the suite,
        storing the loaded models in :attr:`models`.

        :raises FileNotFoundError: If a model file path specified in
                                    :attr:`model_file_names` does not exist.
        """
        for target_name in self.config.get_target_names():
            for method_name, method_obj in self.base_model.method_objs.items():
                method_lower = getattr(method_obj, "short_name", method_name).lower()
                comp_key = f"{method_lower}_{target_name}"

                path = self.model_file_names.get(comp_key)
                if not path or not os.path.exists(path):
                    raise FileNotFoundError(f"File '{path}' does not exist.")

                config_method = copy.deepcopy(self.config)
                config_method.set_base_class("model", method_name)
                new_model_instance = load_single_model_class_with_class_name(
                    config_method, method_name
                )

                new_model_instance.load_model(path)
                new_model_instance = new_model_instance.update_nthreads(
                    new_model_instance
                )

                self.models[comp_key] = new_model_instance

    def create_metric_plots(self) -> None:
        """
        Override parent method to call the multi-method metric plotter.
        """
        create_multi_method_metric_plots(self)
