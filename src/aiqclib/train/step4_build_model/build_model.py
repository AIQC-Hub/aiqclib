"""
This module defines the :class:`BuildModel` class, a specialized component
for building and testing machine learning models.

It inherits from :class:`aiqclib.train.step4_build_model.build_model_base.BuildModelBase`
and orchestrates the training and evaluation of models for specified targets
using Polars DataFrames.
"""

from typing import Optional, Dict

import polars as pl

from aiqclib.common.base.config_base import ConfigBase
from aiqclib.train.step4_build_model.build_model_base import BuildModelBase


class BuildModel(BuildModelBase):
    """
    A subclass of :class:`BuildModelBase` designed to build and test models
    using provided training and test sets for each target.

    This class sets its :attr:`expected_class_name` to ``"BuildModel"``,
    which must match the YAML configuration's ``base_class`` if it is to be
    instantiated within that framework. It extends the base functionality
    to specifically manage training and testing workflows, including
    data preparation steps like column dropping for model input and
    result aggregation.
    """

    expected_class_name: str = "BuildModel"

    def __init__(
        self,
        config: ConfigBase,
        training_sets: Optional[Dict[str, pl.DataFrame]] = None,
        test_sets: Optional[Dict[str, pl.DataFrame]] = None,
    ) -> None:
        """
        Initializes the BuildModel class with a training configuration,
        a dictionary of training sets, and optionally a dictionary
        of test sets.

        :param config: A training configuration object specifying paths,
                       parameters, and model-building directives.
        :type config: ConfigBase
        :param training_sets: A dictionary of training data keyed by target name,
                              each value being a Polars DataFrame. Defaults to None.
        :type training_sets: Optional[Dict[str, pl.DataFrame]]
        :param test_sets: A dictionary of test data keyed by target name,
                          each value being a Polars DataFrame. Defaults to None.
        :type test_sets: Optional[Dict[str, pl.DataFrame]]
        """
        super().__init__(
            config=config, training_sets=training_sets, test_sets=test_sets
        )

        self.drop_cols = ["row_id", "platform_code", "profile_no", "observation_no"]

        self.test_cols = [
            "row_id",
            "platform_code",
            "profile_no",
            "observation_no",
            "label",
        ]

    def build(self, target_name: str) -> None:
        """
        Build (train) a test model for the specified target, storing it in :attr:`models`.

        This method:

          1. Reloads the base model via :meth:`load_base_model`.
          2. Attaches the training data for the target (dropping the ``k_fold`` column
             and common identifying columns).
          3. Calls :meth:`base_model.build`.
          4. Stores the built model in :attr:`models[target_name]`.

        :param target_name: The target variable name, used to index
                            :attr:`training_sets` and locate the training data.
        :type target_name: str
        :raises ValueError: If :attr:`training_sets` is empty,
                            indicating no corresponding data is available for model building.
        """
        if not self.training_sets:
            raise ValueError("Member variable 'training_sets' must not be empty.")

        self.load_base_model()

        self.base_model.training_set = self.training_sets[target_name].drop(
            ["k_fold"] + self.drop_cols
        )
        self.base_model.build()
        self.models[target_name] = self.base_model

    def build_final_model(self, target_name: str) -> None:
        """
        Build (train) a model for the specified target, storing it in :attr:`final_models`.

        This method:

          1. Reloads the base model via :meth:`load_base_model`.
          2. Attaches the training data for the target (dropping the ``k_fold`` column
             and common identifying columns).
          3. Attaches the test data for the target (dropping common identifying columns).
          4. Calls :meth:`base_model.build`.
          5. Stores the built model in :attr:`models[target_name]`.

        :param target_name: The target variable name, used to index
                            :attr:`training_sets` and locate the training data.
        :type target_name: str
        :raises ValueError: If :attr:`training_sets` or :attr:`test_sets` is empty,
                            indicating no corresponding data is available for model building.
        """
        if not self.training_sets:
            raise ValueError("Member variable 'training_sets' must not be empty.")

        if not self.test_sets:
            raise ValueError("Member variable 'test_sets' must not be empty.")

        self.load_base_model()

        training_set = self.training_sets[target_name].drop(["k_fold"] + self.drop_cols)
        test_set = self.test_sets[target_name].drop(self.drop_cols)
        self.base_model.training_set = training_set.vstack(test_set)

        self.base_model.build()
        self.final_models[target_name] = self.base_model

    def test(self, target_name: str) -> None:
        """
        Test the model for the given target, storing the results in :attr:`results`.

        This method:

          1. Retrieves the previously built model from :attr:`models[target_name]`.
          2. **Resets the model's contingency table** to ensure no data duplication
             from previous runs.
          3. Attaches the appropriate test set from :attr:`test_sets[target_name]`,
             dropping common identifying columns.
          4. Calls :meth:`base_model.test`.
          5. Stores the test report in :attr:`reports[target_name]`.
          6. Stores the contingency table in :attr:`contingency_tables[target_name]`.
          7. Stores the SHAP values in :attr:`shap_values[target_name]`.
          8. Stores the test predictions, augmented with identifying information
             and the true label, in :attr:`predictions[target_name]`.

        :param target_name: The target variable name, used to index
                            both :attr:`models` and :attr:`test_sets`.
        :type target_name: str
        """
        self.base_model = self.models[target_name]

        # Reset contingency table to avoid duplication if test is run multiple times
        self.base_model.contingency_table = None

        self.base_model.test_set = self.test_sets[target_name].drop(self.drop_cols)
        self.base_model.test()
        self.reports[target_name] = self.base_model.report

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
