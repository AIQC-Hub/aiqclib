"""
This module defines the ClassifyAll class, a specialized implementation
of BuildModelBase designed for building and testing classification models
across multiple targets. It manages configuration, data handling, and
result persistence for a comprehensive classification workflow.
"""

from typing import Optional, Dict

import polars as pl

from aiqclib.common.base.config_base import ConfigBase
from aiqclib.train.step4_build_model.build_model_base import BuildModelBase


class ClassifyAll(BuildModelBase):
    """
    A subclass of :class:`BuildModelBase` that orchestrates the building
    and testing of classification models for multiple targets using
    provided training and test sets.

    This class sets its :attr:`expected_class_name` to ``"ClassifyAll"``,
    which must match the YAML configuration’s ``base_class`` if you
    intend to instantiate it within that framework.
    """

    expected_class_name: str = "ClassifyAll"

    def __init__(
        self,
        config: ConfigBase,
        test_sets: Optional[Dict[str, pl.DataFrame]] = None,
    ) -> None:
        """
        Initialize the ClassifyAll instance.

        This constructor configures the classification process,
        setting up file names for models and predictions, and identifying
        columns to be dropped or kept during processing.

        .. warning::
           This class specifically initializes the parent :class:`BuildModelBase`
           with ``training_sets=None``, implying that the actual training
           data fetching/loading might be handled differently or by the
           underlying base model once it's set.

        :param config: A training configuration object specifying paths,
                       parameters, and model-building directives.
        :type config: ConfigBase
        :param test_sets: A dictionary of test data keyed by target name,
                          each value being a Polars DataFrame. Defaults to None.
        :type test_sets: Optional[Dict[str, pl.DataFrame]]
        """
        super().__init__(
            config=config, training_sets=None, test_sets=test_sets, step_name="classify"
        )

        #: Default names for model files and test reports,
        #: with placeholders for the target name.
        self.default_file_names: Dict[str, str] = {
            "report": "classify_report_{target_name}.tsv",
            "prediction": "classify_prediction_{target_name}.parquet",
            "contingency_table": "classify_contingency_tables_{target_name}.parquet",
            "shap_value": "classify_shap_values_{target_name}.parquet",
            "metric_plot": "classify_metric_plots_{target_name}.svg",
        }
        self.default_model_file_name: str = "model_{target_name}.joblib"

        #: A dictionary mapping "model" or "result" to
        #: target-specific file paths, derived from configuration.
        self.output_file_names: Dict[str, Dict[str, str]] = {
            k: self.config.get_target_file_names("classify", v)
            for k, v in self.default_file_names.items()
        }

        #: A dictionary mapping "model" to target-specific file paths,
        #: derived from configuration.
        self.model_file_names: Dict[str, str] = self.config.get_target_file_names(
            step_name="model",
            default_file_name=self.default_model_file_name,
            use_dataset_folder=False,
        )

        #: Columns to be dropped from the test set before passing to the base model.
        self.drop_cols = ["row_id", "platform_code", "profile_no", "observation_no"]

        #: Columns to be selected from the original test set for final prediction output.
        self.test_cols = [
            "row_id",
            "platform_code",
            "profile_no",
            "observation_no",
            "label",
        ]

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
        Test the model for the given target, storing the results in :attr:`results`.

        This method performs the following steps:

          1. Retrieves the trained model from :attr:`models[target_name]`.
          2. **Resets the model's contingency table** to ensure no data duplication
             from previous runs.
          3. Prepares the appropriate test set by dropping specified columns
             from :attr:`test_sets[target_name]` and attaches it to the
             :attr:`base_model`.
          4. Calls the :meth:`base_model.test` method to generate predictions and reports.
          5. Stores the contingency table in :attr:`contingency_tables[target_name]`.
          6. Concatenates relevant original test set columns with the
             generated predictions and stores them in :attr:`predictions[target_name]`.
          7. Stores the test report from the base model in :attr:`reports[target_name]`.

        :param target_name: The target variable name, used to index
                            both :attr:`models` and :attr:`test_sets`.
        :type target_name: str
        """
        self.base_model = self.models[target_name]

        # Reset contingency table to avoid duplication if test is run multiple times
        self.base_model.contingency_table = None

        self.base_model.test_set = self.test_sets[target_name].drop(self.drop_cols)
        self.base_model.test()

        if self.base_model.contingency_table is not None:
            self.contingency_tables[target_name] = self.base_model.contingency_table

        if self.base_model.shap_values is not None:
            self.shap_values[target_name] = self.base_model.shap_values

        predictions = self.base_model.predictions
        self.predictions[target_name] = pl.concat(
            [
                self.test_sets[target_name].select(self.test_cols),
                predictions,
            ],
            how="horizontal",
        )
        self.reports[target_name] = self.base_model.report
