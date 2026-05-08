"""
This module provides the KFoldValidationSuite class, an implementation of
k-fold cross-validation tailored for validating multiple ML algorithms
simultaneously via the ModelSuite class.
"""

from typing import Optional, List, Dict
import copy

import polars as pl

from aiqclib.common.base.config_base import ConfigBase
from aiqclib.train.step2_validate_model.validate_base import ValidationBase


class KFoldValidationSuite(ValidationBase):
    """
    A subclass of :class:`ValidationBase` that performs k-fold cross-validation
    on training sets across multiple machine learning methods provided by a
    model suite (e.g., :class:`ModelSuite`).

    This class iterates over the specified number of folds and across all
    methods defined in the base model. Results are accumulated with composite
    keys (method + target) to ensure outputs are saved uniquely per method.
    """

    expected_class_name: str = "KFoldValidationSuite"

    def __init__(
        self,
        config: ConfigBase,
        training_sets: Optional[Dict[str, pl.DataFrame]] = None,
    ) -> None:
        """
        Initialize the k-fold validation suite process.

        :param config: A training configuration object containing
                       model parameters, file paths, and other
                       validation settings.
        :type config: ConfigBase
        :param training_sets: A dictionary where keys are target names and values are
                              Polars DataFrames of labeled data. Each DataFrame must
                              contain a column named ``k_fold``. Defaults to None.
        :type training_sets: Optional[Dict[str, pl.DataFrame]]
        :raises ValueError: If the configured base model does not have the `multi`
                            flag set to True.
        """
        super().__init__(config=config, training_sets=training_sets)

        # Ensure the base model is a multi-method suite (like ModelSuite)
        if not getattr(self.base_model, "multi", False):
            raise ValueError(
                "KFoldValidationSuite requires a base model with 'multi=True' "
                "(e.g., ModelSuite), but received a standard model class."
            )

        # Redefine default file names to include the {method} placeholder
        self.default_file_names: Dict[str, str] = {
            "report": "validation_report_{method}_{target_name}.tsv",
            "contingency_table": "contingency_tables_{method}_{target_name}.parquet",
            "metric_plot": "metric_plots_{method}_{target_name}.svg",
        }

        # Re-generate output file names using the new pattern with {method}
        # For each output type (report, contingency_table, metric_plot),
        # get the target-specific filenames from config.
        # These filenames will still contain the {method} placeholder,
        # which will be replaced later in the validate method for each specific method.
        self.output_file_names: Dict[str, Dict[str, str]] = {
            k: self.config.get_target_file_names(
                step_name="validate", default_file_name=v
            )
            for k, v in self.default_file_names.items()
        }

        #: The default number of folds if none is specified in the config.
        self.default_k_fold: int = 10
        self.drop_cols = [
            "k_fold",
            "row_id",
            "platform_code",
            "profile_no",
            "observation_no",
        ]

        self.base_model.set_enable_shap(False)

    def get_k_fold(self) -> int:
        """
        Retrieve the number of folds to use for cross-validation from
        the ``validate`` section of the YAML config, or fall back
        to :attr:`default_k_fold`.

        :return: The number of folds for k-fold cross-validation.
        :rtype: int
        """
        return (
            self.config.get_step_params("validate").get("k_fold", self.default_k_fold)
            or self.default_k_fold
        )

    def validate(self, target_name: str) -> None:
        """
        Conduct k-fold cross-validation for the given target name across all
        methods in the ModelSuite.

        For each method in ``base_model.method_objs``:
          1. Iterate over the defined number of folds.
          2. Build the model using all training data except the current fold.
          3. Test the model on the held-out fold.
          4. Accumulate test results and contingency tables under a composite
             key (`{method_name}_{target_name}`).
          5. Update `output_file_names` to replace the `{method}` placeholder.

        :param target_name: The identifier for which target dataset to validate.
        :type target_name: str
        """
        k_fold: int = self.get_k_fold()

        # Iterate through all configured ML methods loaded in ModelSuite
        for method_name, method_obj in self.base_model.method_objs.items():
            # Retrieve short_name (fallback to method_name if attribute is missing), and lowercase it
            method_lower = getattr(method_obj, "short_name", method_name).lower()

            # Create a composite key (e.g. "xgb_temp") to uniquely store results
            # and map to the parent ValidationBase's dictionaries.
            comp_key = f"{method_lower}_{target_name}"

            self.models[comp_key] = []
            reports: List[pl.DataFrame] = []
            contingency_tables: List[pl.DataFrame] = []

            for k in range(k_fold):
                # We need a fresh copy of the specific ML method model for each fold
                current_fold_model = copy.deepcopy(method_obj)
                current_fold_model.k = k + 1

                current_fold_model.training_set = (
                    self.training_sets[target_name]
                    .filter(pl.col("k_fold") != (k + 1))
                    .drop(self.drop_cols)
                )
                current_fold_model.build()
                self.models[comp_key].append(current_fold_model)

                current_fold_model.test_set = (
                    self.training_sets[target_name]
                    .filter(pl.col("k_fold") == (k + 1))
                    .drop(self.drop_cols)
                )
                current_fold_model.test()
                reports.append(current_fold_model.report)

                if current_fold_model.contingency_table is not None:
                    contingency_tables.append(current_fold_model.contingency_table)

            # Store the aggregated results using the composite key
            self.reports[comp_key] = pl.concat(reports)

            if contingency_tables:
                self.contingency_tables[comp_key] = pl.concat(contingency_tables)

            # Resolve the {method} placeholder in the output file paths specifically for this composite key.
            # The original target_name entry in self.output_file_names still contains the {method} placeholder.
            # This creates a new entry for the composite key with the resolved path.
            self.output_file_names["report"][comp_key] = self.output_file_names[
                "report"
            ][target_name].replace("{method}", method_lower)
            self.output_file_names["contingency_table"][comp_key] = (
                self.output_file_names["contingency_table"][target_name].replace(
                    "{method}", method_lower
                )
            )
            self.output_file_names["metric_plot"][comp_key] = self.output_file_names[
                "metric_plot"
            ][target_name].replace("{method}", method_lower)
