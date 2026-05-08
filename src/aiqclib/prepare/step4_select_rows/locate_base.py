"""
This module defines the abstract base class :class:`LocatePositionBase` for identifying
and extracting specific rows from a dataset based on defined target criteria.

It provides a structured approach for processing different targets, typically
for purposes like creating training datasets or selecting specific data subsets,
leveraging configuration settings and handling data I/O.
"""

import os
from abc import abstractmethod
from typing import Dict, Optional

import polars as pl

from aiqclib.common.base.config_base import ConfigBase
from aiqclib.common.base.dataset_base import DataSetBase


class LocatePositionBase(DataSetBase):
    """
    Abstract base class for locating and extracting target rows from a dataset.

    This class extends :class:`aiqclib.common.base.dataset_base.DataSetBase` to validate
    that the YAML configuration matches the expected structure and to provide a framework
    for operations related to identifying rows of interest (e.g., training data).
    Subclasses must implement:

    - The :meth:`locate_target_rows` method for per-target row identification logic.
    - Potentially define ``expected_class_name`` if this class is intended to be
      directly instantiated and matched against the YAML's ``base_class`` configuration.
    """

    def __init__(
        self,
        config: ConfigBase,
        input_data: Optional[pl.DataFrame] = None,
        selected_profiles: Optional[pl.DataFrame] = None,
    ) -> None:
        """
        Initialize the base class for locating position or training rows within a dataset.

        Initializes the class by calling the constructor of :class:`DataSetBase` and
        setting up member variables to store input data, selected profiles, and
        the resulting target rows, along with output file path configurations.

        :param config: Configuration object for dataset paths and target definitions.
        :type config: :class:`aiqclib.common.base.config_base.ConfigBase`
        :param input_data: A Polars DataFrame containing the full dataset from which
                           target rows can be extracted, defaults to None.
        :type input_data: :class:`polars.DataFrame`, optional
        :param selected_profiles: A Polars DataFrame containing pre-selected profiles
                                  or rows, if applicable, defaults to None. This can
                                  be used to filter the ``input_data`` further.
        :type selected_profiles: :class:`polars.DataFrame`, optional
        :raises NotImplementedError: If ``expected_class_name`` is not defined
                                     by a subclass and this class is directly instantiated.
        :raises ValueError: If the YAML's ``base_class`` does not match
                            the subclass's ``expected_class_name``.
        """
        super().__init__(step_name="locate", config=config)

        #: str: Default file name template for writing target rows (one file per target).
        #: The ``{target_name}`` placeholder will be replaced.
        self.default_file_name: str = "selected_rows_{target_name}.parquet"

        #: Dict[str, str]: Dictionary mapping each target name to the corresponding
        #: output Parquet file path derived from the configuration.
        self.output_file_names: Dict[str, str] = self.config.get_target_file_names(
            step_name="locate", default_file_name=self.default_file_name
        )

        #: Optional[:class:`polars.DataFrame`]: An optional Polars DataFrame from which
        #: target rows will be extracted. This is the primary input dataset.
        self.input_data: Optional[pl.DataFrame] = input_data

        #: Optional[:class:`polars.DataFrame`]: An optional Polars DataFrame of
        #: pre-selected profiles or rows that might be combined with the input data
        #: during the target-location process, or used as a filter.
        self.selected_profiles: Optional[pl.DataFrame] = selected_profiles

        #: Dict[str, :class:`polars.DataFrame`]: A dictionary to store the resulting
        #: target rows for each target as a Polars DataFrame, keyed by target name.
        self.selected_rows: Dict[str, pl.DataFrame] = {}

    def process_targets(self) -> None:
        """
        Iterate over all defined targets and call :meth:`locate_target_rows` on each.

        This method retrieves the target definitions (names and other metadata) from
        the configuration object (:attr:`config`) and then sequentially processes
        each target. The concrete logic for identifying rows per target is
        implemented in subclasses via the abstract :meth:`locate_target_rows` method.
        """
        for target_name, target_info in self.config.get_target_dict().items():
            self.locate_target_rows(target_name, target_info)

    @abstractmethod
    def locate_target_rows(self, target_name: str, target_value: Dict) -> None:
        """
        Abstract method to locate rows in :attr:`input_data` or :attr:`selected_profiles`
        relevant to a specific target.

        Subclasses must implement this method to define the specific logic for
        identifying and extracting the subset of rows matching the criteria
        defined by the target. The identified rows for the given target should
        be stored in the :attr:`selected_rows` dictionary under the ``target_name`` key.

        :param target_name: The name of the target variable (e.g., 'training_data').
        :type target_name: str
        :param target_value: A dictionary containing metadata or specific criteria
                             for the target, as defined in the configuration.
        :type target_value: Dict
        """
        pass  # pragma: no cover

    def write_selected_rows(self) -> None:
        """
        Write the identified target rows to separate Parquet files.

        This method iterates through the :attr:`selected_rows` dictionary. For each
        target, it constructs the output file path using the template defined
        in :attr:`output_file_names` and writes the corresponding Polars DataFrame
        to a Parquet file. Directories are created if they do not exist.

        :raises ValueError: If the :attr:`selected_rows` dictionary is empty, indicating
                            that no target rows have been identified or processed.
        """
        if not self.selected_rows:
            raise ValueError(
                "Member variable 'selected_rows' must not be empty. "
                "Please ensure 'process_targets' has been called and "
                "data has been located."
            )

        for target_name, df in self.selected_rows.items():
            file_path = self.output_file_names[target_name]
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            df.write_parquet(file_path)
