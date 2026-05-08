"""
This module defines the abstract base class `SplitDataSetBase` for managing
the splitting of target feature DataFrames into training and test sets,
and for assigning k-fold cross-validation labels.

It extends `DataSetBase` to provide a standardized structure for data splitting
operations, integrating configuration management and supporting the output
of processed datasets to Parquet files.
"""

import os
from abc import abstractmethod
from typing import Dict, Optional

import polars as pl

from aiqclib.common.base.config_base import ConfigBase
from aiqclib.common.base.dataset_base import DataSetBase


class SplitDataSetBase(DataSetBase):
    """
    Abstract base class to perform train/test splitting and k-fold assignment
    for target feature DataFrames.

    This class extends :class:`aiqclib.common.base.dataset_base.DataSetBase`
    to validate and incorporate YAML-based configuration. It provides methods
    for writing out the resulting training and test sets into Parquet files.

    Subclasses must implement the abstract methods:
    :meth:`split_test_set`, :meth:`add_k_fold`, and :meth:`drop_columns`.

    .. note::

       Since this class inherits from :class:`aiqclib.common.base.dataset_base.DataSetBase`
       and is marked as an abstract base class, it may require an ``expected_class_name``
       defined by subclasses if they are intended to be instantiated.
    """

    def __init__(
        self,
        config: ConfigBase,
        target_features: Optional[Dict[str, pl.DataFrame]] = None,
    ) -> None:
        """
        Initialize the train/test splitting class with a configuration
        and optional target features.

        :param config: A dataset configuration object containing parameters
                       and paths for splitting.
        :type config: aiqclib.common.base.config_base.ConfigBase
        :param target_features: A dictionary where keys are target names (str)
                                and values are Polars DataFrames holding combined
                                features for each target, or None if not yet available.
        :type target_features: Optional[Dict[str, polars.DataFrame]]

        :raises NotImplementedError: If ``expected_class_name`` is not set in a subclass
                                     and an instance is directly created.
        :raises ValueError: If the YAML's ``base_class`` does not match
                            the subclass's ``expected_class_name``.
        """
        super().__init__(step_name="split", config=config)

        #: Default file naming templates for train and test sets.
        self.default_file_names: Dict[str, str] = {
            "train": "train_set_{target_name}.parquet",
            "test": "test_set_{target_name}.parquet",
        }
        #: File paths for each target's train/test sets, keyed by "train" and "test".
        self.output_file_names: Dict[str, Dict[str, str]] = {
            k: self.config.get_target_file_names(step_name="split", default_file_name=v)
            for k, v in self.default_file_names.items()
        }

        #: A dictionary of Polars DataFrames of feature columns for all targets, if available.
        self.target_features: Optional[Dict[str, pl.DataFrame]] = target_features
        #: A dictionary of Polars DataFrames holding training splits by target name.
        self.training_sets: Dict[str, pl.DataFrame] = {}
        #: A dictionary of Polars DataFrames holding test splits by target name.
        self.test_sets: Dict[str, pl.DataFrame] = {}

        #: Default fraction for test sets if none is specified in the config.
        self.default_test_set_fraction: float = 0.1
        #: Default number of folds for k-fold cross-validation if unspecified.
        self.default_k_fold: int = 10

    def get_test_set_fraction(self) -> float:
        """
        Retrieve the test set fraction (0-1) from configuration or fallback.

        :return: A float in the range [0, 1] representing the fraction of data
                  reserved for testing.
        :rtype: float
        """
        return (
            self.config.get_step_params("split").get(
                "test_set_fraction", self.default_test_set_fraction
            )
            or self.default_test_set_fraction
        )

    def get_k_fold(self) -> int:
        """
        Retrieve the number of folds for cross-validation from configuration or fallback.

        :return: An integer representing how many folds are used during k-fold
                  cross-validation steps.
        :rtype: int
        """
        return (
            self.config.get_step_params("split").get("k_fold", self.default_k_fold)
            or self.default_k_fold
        )

    def process_targets(self) -> None:
        """
        Perform test splitting, k-fold assignment, and column dropping
        for each target defined in the dataset configuration.

        Uses the abstract methods :meth:`split_test_set`, :meth:`add_k_fold`,
        and :meth:`drop_columns` for each target name.
        """
        for target_name in self.config.get_target_names():
            self.split_test_set(target_name)
            self.add_k_fold(target_name)
            self.drop_columns(target_name)

    @abstractmethod
    def split_test_set(self, target_name: str) -> None:
        """
        Split the DataFrame for a given target into training and test sets.

        Must store any resulting DataFrames in :attr:`training_sets`
        and :attr:`test_sets` using the target name as a key.

        :param target_name: The identifier of the target to split.
        :type target_name: str
        """
        pass  # pragma: no cover

    @abstractmethod
    def add_k_fold(self, target_name: str) -> None:
        """
        Add k-fold cross-validation columns or labels to the training set.

        Typically, this method would modify the DataFrame in
        :attr:`training_sets[target_name]`.

        :param target_name: The target name being processed.
        :type target_name: str
        """
        pass  # pragma: no cover

    @abstractmethod
    def drop_columns(self, target_name: str) -> None:
        """
        Drop unnecessary columns from both training and test sets.

        :param target_name: The target name being processed.
        :type target_name: str
        """
        pass  # pragma: no cover

    def write_training_sets(self) -> None:
        """
        Write the training splits to Parquet files.

        :raises ValueError: If :attr:`training_sets` is empty (i.e.,
                            no splits have been created).
        """
        if not self.training_sets:
            raise ValueError("Member variable 'training_sets' must not be empty.")

        for target_name, df in self.training_sets.items():
            output_path = self.output_file_names["train"][target_name]
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df.write_parquet(output_path)

    def write_test_sets(self) -> None:
        """
        Write the test splits to Parquet files.

        :raises ValueError: If :attr:`test_sets` is empty (i.e.,
                            no splits have been created).
        """
        if not self.test_sets:
            raise ValueError("Member variable 'test_sets' must not be empty.")

        for target_name, df in self.test_sets.items():
            output_path = self.output_file_names["test"][target_name]
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df.write_parquet(output_path)

    def write_data_sets(self) -> None:
        """
        Write both training and test sets to disk.

        Simply calls :meth:`write_test_sets` and :meth:`write_training_sets`.
        """
        self.write_test_sets()
        self.write_training_sets()
