"""
This module defines the InputTrainingSetBase class, which serves as a base for
importing pre-split training and test datasets. It leverages a training-specific
configuration to identify and load Parquet files into Polars DataFrames,
managing both training and test sets for multiple targets.
"""

import os
from typing import Dict

import polars as pl

from aiqclib.common.base.config_base import ConfigBase
from aiqclib.common.base.dataset_base import DataSetBase


class InputTrainingSetBase(DataSetBase):
    """
    A base class for importing pre-split training and test data sets,
    leveraging the training-specific configuration (:class:`ConfigBase`).

    This class extends :class:`DataSetBase` to ensure that the given YAML
    configuration is valid for the step named ``"input"``.
    It provides logic for iterating over targets, identifying the Parquet
    files for each, and reading them into memory.

    .. note::

       Since this class inherits from :class:`DataSetBase`, a subclass
       or this class itself may need to define an ``expected_class_name``
       that matches the YAML's ``base_class`` if you plan to instantiate it
       directly. Otherwise, :class:`DataSetBase` may raise a
       :class:`NotImplementedError`.
    """

    def __init__(self, config: ConfigBase) -> None:
        """
        Initialize the training data set importer with a training configuration.

        :param config: A training configuration instance that includes
                       file paths and target definitions.
        :type config: aiqclib.common.base.config_base.ConfigBase
        :raises NotImplementedError: If no ``expected_class_name``
                                     is defined by a subclass and this
                                     class is instantiated directly
                                     (per :class:`DataSetBase` logic).
        :raises ValueError: If the provided config's ``base_class``
                            does not match  ``expected_class_name``
                            (also per :class:`DataSetBase`).
        """
        super().__init__(step_name="input", config=config)

        #: Default file naming patterns for train/test sets. The substring
        #: ``{target_name}`` will be replaced dynamically.
        self.default_file_names: Dict[str, str] = {
            "train": "train_set_{target_name}.parquet",
            "test": "test_set_{target_name}.parquet",
        }

        #: A mapping of "train" and "test" to dictionaries of target-specific file names.
        #:
        #: Example format::
        #:
        #:     {
        #:         "train": {"targetA": "path/to/targetA_train.parquet", ...},
        #:         "test":  {"targetA": "path/to/targetA_test.parquet", ...}
        #:     }
        self.input_file_names: Dict[str, Dict[str, str]] = {
            k: self.config.get_target_file_names(step_name="input", default_file_name=v)
            for k, v in self.default_file_names.items()
        }

        #: A dictionary mapping target names to Polars DataFrames
        #: containing their training set.
        self.training_sets: Dict[str, pl.DataFrame] = {}
        #: A dictionary mapping target names to Polars DataFrames
        #: containing their test set.
        self.test_sets: Dict[str, pl.DataFrame] = {}

    def process_targets(self) -> None:
        """
        Iterate over all targets defined in the config and read both
        training and test sets from Parquet files.

        Utilizes :meth:`read_training_set` and :meth:`read_test_sets`
        for each target name returned by
        :meth:`~aiqclib.common.base.config_base.ConfigBase.get_target_names`.
        """
        for target_name in self.config.get_target_names():
            self.read_training_set(target_name)
            self.read_test_sets(target_name)

    def read_training_set(self, target_name: str) -> None:
        """
        Read a single target-specific training set from a Parquet file
        into :attr:`training_sets`.

        :param target_name: The identifier of the target dataset to be loaded.
        :type target_name: str
        :raises FileNotFoundError: If the corresponding Parquet file does not exist.
        """
        file_name: str = self.input_file_names["train"][target_name]
        if not os.path.exists(file_name):
            raise FileNotFoundError(f"File '{file_name}' does not exist.")
        self.training_sets[target_name] = pl.read_parquet(file_name)

    def read_test_sets(self, target_name: str) -> None:
        """
        Read a single target-specific test set from a Parquet file
        into :attr:`test_sets`.

        :param target_name: The identifier of the target dataset to be loaded.
        :type target_name: str
        :raises FileNotFoundError: If the corresponding Parquet file does not exist.
        """
        file_name: str = self.input_file_names["test"][target_name]
        if not os.path.exists(file_name):
            raise FileNotFoundError(f"File '{file_name}' does not exist.")
        self.test_sets[target_name] = pl.read_parquet(file_name)
