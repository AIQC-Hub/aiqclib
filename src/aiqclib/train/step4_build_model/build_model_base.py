"""
Provides an abstract base class, :class:`aiqclib.common.base.build_model_base.BuildModelBase`,
for building and testing machine learning models using structured training and test datasets.

This module establishes a framework for model development within a larger
data quality control (DMQC) system, integrating with configuration management
and model loading utilities. Subclasses are expected to implement specific
model building and testing logic tailored to different modeling paradigms or
frameworks.
"""

import os
from abc import abstractmethod
from typing import Optional, Dict, Any

import polars as pl

from aiqclib.common.base.config_base import ConfigBase
from aiqclib.common.base.dataset_base import DataSetBase
from aiqclib.common.base.model_base import ModelBase
from aiqclib.common.loader.model_loader import load_model_class
from aiqclib.common.utils.metric_plots import create_metric_plots


class BuildModelBase(DataSetBase):
    """
    An abstract base class to build and test models, using training/test sets
    and a YAML-based configuration.

    Inherits from :class:`aiqclib.common.base.dataset_base.DataSetBase` (with step name ``"build"``)
    to ensure that the provided configuration matches the expected
    fields for model-building. Subclasses must define their own
    logic in the :meth:`build` and :meth:`test` abstract methods,
    potentially for different modeling frameworks.
    """

    def __init__(
        self,
        config: ConfigBase,
        training_sets: Optional[Dict[str, pl.DataFrame]] = None,
        test_sets: Optional[Dict[str, pl.DataFrame]] = None,
        step_name: str = "build",
    ) -> None:
        """
        Initialize the model-building base class with optional training
        and test sets.

        :param config: A training configuration object containing
                       paths and parameters for building and testing models.
        :type config: aiqclib.common.base.config_base.ConfigBase
        :param training_sets: A dictionary of :class:`polars.DataFrame`, where keys are target
                              names and values are DataFrames with training examples
                              for that target. Defaults to :obj:`None`.
        :type training_sets: Optional[Dict[str, polars.DataFrame]]
        :param test_sets: A dictionary of :class:`polars.DataFrame`, where keys are target
                          names and values are DataFrames with testing examples
                          for that target. Defaults to :obj:`None`.
        :type test_sets: Optional[Dict[str, polars.DataFrame]]
        :param step_name: The name of the current processing step,
                          defaults to "build".
        :type step_name: str
        """
        super().__init__(step_name=step_name, config=config)

        #: Default names for model files and test reports,
        #: with placeholders for the target name.
        self.default_file_names: Dict[str, str] = {
            "report": "test_report_{target_name}.tsv",
            "prediction": "test_prediction_{target_name}.parquet",
            "contingency_table": "test_contingency_tables_{target_name}.parquet",
            "shap_value": "test_shap_values_{target_name}.parquet",
            "metric_plot": "test_metric_plots_{target_name}.svg",
        }
        self.default_model_file_name: str = "model_{target_name}.joblib"

        #: A dictionary mapping result type (e.g., "report", "prediction") to
        #: target-specific file paths.
        self.output_file_names: Dict[str, Dict[str, str]] = {
            k: self.config.get_target_file_names(step_name="build", default_file_name=v)
            for k, v in self.default_file_names.items()
        }

        #: A dictionary mapping "model" to target-specific file paths.
        self.model_file_names: Dict[str, str] = self.config.get_target_file_names(
            "model", self.default_model_file_name
        )

        #: A dictionary containing training data keyed by target name.
        self.training_sets: Optional[Dict[str, pl.DataFrame]] = training_sets
        #: A dictionary containing test data keyed by target name.
        self.test_sets: Optional[Dict[str, pl.DataFrame]] = test_sets

        #: The base model instance loaded from :meth:`load_base_model`;
        #: can be overridden for each target.
        self.base_model: Optional[ModelBase] = None
        self.load_base_model()

        #: A dictionary to store model objects keyed by target name.
        self.models: Dict[str, Optional[ModelBase]] = {}
        self.final_models: Dict[str, Optional[ModelBase]] = {}

        #: A dictionary to store test reports keyed by target name.
        self.reports: Dict[str, pl.DataFrame] = {}
        #: A dictionary to store contingency tables keyed by target name.
        self.contingency_tables: Dict[str, pl.DataFrame] = {}
        #: A dictionary to store SHAP values keyed by target name.
        self.shap_values: Dict[str, pl.DataFrame] = {}
        #: A dictionary to store prediction results keyed by target name.
        self.predictions: Dict[str, pl.DataFrame] = {}

    def load_base_model(self) -> None:
        """
        Load the base model class from the configuration.

        The loaded model is stored in :attr:`base_model` and may be cloned,
        specialized, or reloaded for each target in the building process.
        """
        self.base_model = load_model_class(self.config)

    def build_final_model_targets(self) -> None:
        """
        Iterate over all targets from the configuration, calling :meth:`build_final_model`
        for each target.
        """
        for target_name in self.config.get_target_names():
            self.build_final_model(target_name)

    def build_targets(self) -> None:
        """
        Iterate over all targets from the configuration, calling :meth:`build_test`
        for each target.
        """
        for target_name in self.config.get_target_names():
            self.build(target_name)

    def test_targets(self) -> None:
        """
        Iterate over all targets, ensuring that a model has been built before
        calling :meth:`test`.

        :raises ValueError: If a target has no corresponding entry in
                            :attr:`models`.
        """
        for target_name in self.config.get_target_names():
            if target_name not in self.models:
                raise ValueError(
                    f"No valid model found for the variable '{target_name}'."
                )
            self.test(target_name)

    @abstractmethod
    def build(self, target_name: str) -> None:
        """
        Build a test model for the specified target name.

        This abstract method must be implemented by subclasses to
        perform the steps necessary for initializing, training,
        and storing the model in :attr:`models`.

        :param target_name: The identifier for this target's model
                            in :attr:`training_sets`.
        :type target_name: str
        """
        pass  # pragma: no cover

    @abstractmethod
    def build_final_model(self, target_name: str) -> None:
        """
        Build a final model for the specified target name.

        This abstract method must be implemented by subclasses to
        perform the steps necessary for initializing, training,
        and storing the model in :attr:`final_models`.

        :param target_name: The identifier for this target's model
                            in :attr:`training_sets`.
        :type target_name: str
        """
        pass  # pragma: no cover

    @abstractmethod
    def test(self, target_name: str) -> None:
        """
        Test a model for the specified target name.

        Typically, this includes running predictions, evaluating
        performance metrics, and storing results in :attr:`reports`.

        :param target_name: The identifier for this target's model
                            and test set in :attr:`test_sets` (plus
                            entries in :attr:`models`).
        :type target_name: str
        """
        pass  # pragma: no cover

    def write_reports(self) -> None:
        """
        Write each target's test reports to a TSV file.

        :raises ValueError: If :attr:`reports` is empty, indicating no tests
                            have been carried out or no reports stored.
        """
        if not self.reports:
            raise ValueError("Member variable 'reports' must not be empty.")

        for target_name, df in self.reports.items():
            output_path = self.output_file_names["report"][target_name]
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df.write_csv(output_path, separator="\t")

    def write_contingency_tables(self) -> None:
        """
        Write each target's contingency table to a Parquet file.

        :raises ValueError: If :attr:`contingency_tables` is empty, indicating no tests
                            have been carried out or no tables stored.
        """
        if not self.contingency_tables:
            raise ValueError("Member variable 'contingency_tables' must not be empty.")

        for target_name, df in self.contingency_tables.items():
            output_path = self.output_file_names["contingency_table"][target_name]
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df.write_parquet(output_path)

    def write_shap_values(self) -> None:
        """
        Write each target's SHAP values to a Parquet file.

        This method checks if SHAP values are enabled in the base model. If not,
        it returns without writing.

        :raises ValueError: If :attr:`shap_values` is empty while SHAP is enabled,
                            indicating no SHAP values were computed or stored.
        """
        if not self.base_model or not getattr(self.base_model, "enable_shap", False):
            # If base_model is None or does not have enable_shap, or enable_shap is False, skip.
            return

        if not self.shap_values:
            raise ValueError(
                "Member variable 'shap_values' must not be empty if SHAP is enabled."
            )

        for target_name, df in self.shap_values.items():
            output_path = self.output_file_names["shap_value"][target_name]
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df.write_parquet(output_path)

    def create_metric_plots(self) -> None:
        """
        Create and save ROC and Precision-Recall plots as an SVG file for each target.

        Calls the common utility function :func:`aiqclib.common.utils.metric_plots.create_metric_plots`.
        """
        create_metric_plots(self)

    def write_models(self) -> None:
        """
        Serialize and write each target's model to disk.

        :raises ValueError: If :attr:`models` is empty, indicating no models
                            have been built for writing.
        """
        if not self.final_models:
            raise ValueError("Member variable 'final_models' must not be empty.")

        for target_name, model_ref in self.final_models.items():
            output_path = self.model_file_names[target_name]
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            if model_ref:
                model_ref.save_model(output_path)

    def read_models(self) -> None:
        """
        Read and restore each target's model from disk, storing
        the loaded model in :attr:`models`.

        :raises FileNotFoundError: If a model file does not exist
                                   for a particular target.
        :raises RuntimeError: If the :attr:`base_model` is not loaded, which is
                              required to update model thread settings.
        """
        for target_name, path in self.model_file_names.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"File '{path}' does not exist.")

            # Assuming load_model_class returns an instance of ModelBase or a compatible type
            new_model_instance: Any = load_model_class(self.config)
            new_model_instance.load_model(path)

            if self.base_model is None:
                raise RuntimeError(
                    "Base model is not loaded; cannot update thread settings for read model."
                )

            new_model_instance = self.base_model.update_nthreads(new_model_instance)
            self.models[target_name] = new_model_instance

    def write_predictions(self) -> None:
        """
        Serialize and write each target's predictions to disk.

        :raises ValueError: If :attr:`predictions` is empty, indicating no predictions
                            have been built for writing.
        """
        if not self.predictions:
            raise ValueError("Member variable 'predictions' must not be empty.")

        for target_name, df in self.predictions.items():
            output_path = self.output_file_names["prediction"][target_name]
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df.write_parquet(output_path)
