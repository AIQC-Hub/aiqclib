"""
Module for profile selection and group labeling base class.

This module defines the abstract base class :class:`ProfileSelectionBase`, which
serves as a template for selecting and labeling profiles within the aiqclib
framework. It integrates with configuration and dataset base classes to
standardize the profile selection step and output.
"""

import os
from abc import abstractmethod
from typing import Optional

import polars as pl

from aiqclib.common.base.config_base import ConfigBase
from aiqclib.common.base.dataset_base import DataSetBase


class ProfileSelectionBase(DataSetBase):
    """
    Abstract base class for profile selection and group labeling.

    Inherits from :class:`aiqclib.common.base.dataset_base.DataSetBase` to leverage
    configuration handling and validation. Subclasses must implement the concrete
    logic for profile labeling.

    Subclasses must define:

    *   ``expected_class_name`` (a class attribute) if they are intended to be
        instantiated (otherwise an error is raised by the base class).
    *   A custom :meth:`label_profiles` method that implements the specific profile
        selection and labeling logic.

    :ivar default_file_name: The default file name for selected profiles.
    :vartype default_file_name: str
    :ivar output_file_name: The full path and name of the output Parquet file
                            where selected profiles will be written.
    :vartype output_file_name: str
    :ivar input_data: An optional Polars DataFrame used as initial data for
                      profile selection.
    :vartype input_data: Optional[polars.DataFrame]
    :ivar selected_profiles: A Polars DataFrame containing the profiles after
                             selection and labeling, typically including a
                             "group_label" column.
    :vartype selected_profiles: Optional[polars.DataFrame]
    """

    def __init__(
        self, config: ConfigBase, input_data: Optional[pl.DataFrame] = None
    ) -> None:
        """
        Initialize the profile selection base class with configuration and optional input data.

        This constructor calls the base class :meth:`aiqclib.common.base.dataset_base.DataSetBase.__init__`
        and sets up file paths and data holders.

        :param config: A dataset configuration object that provides file naming
                       conventions and folder paths.
        :type config: aiqclib.common.base.config_base.ConfigBase
        :param input_data: Optional Polars DataFrame that serves as the initial data
                           for profile selection, defaults to None.
        :type input_data: Optional[polars.DataFrame]
        :raises NotImplementedError: If ``expected_class_name`` is not defined by a subclass
                                      when the base class constructor is called.
        :raises ValueError: If the "base_class" field in the YAML configuration
                            does not match the subclass's ``expected_class_name``.
        """
        super().__init__(step_name="select", config=config)

        self.default_file_name: str = "selected_profiles.parquet"
        self.output_file_name: str = self.config.get_full_file_name(
            step_name="select", default_file_name=self.default_file_name
        )
        self.input_data: Optional[pl.DataFrame] = input_data
        self.selected_profiles: Optional[pl.DataFrame] = None

    @abstractmethod
    def label_profiles(self) -> None:
        """
        Abstract method to be implemented by subclasses for labeling profiles
        to identify positive and negative groups.

        Implementations of this method should perform the core logic for
        profile selection and labeling, assigning the resulting DataFrame
        (which should typically include a 'group_label' column) to the
        :attr:`selected_profiles` instance variable.
        """
        pass  # pragma: no cover

    def write_selected_profiles(self) -> None:
        """
        Write the selected profiles to a Parquet file.

        The output file path is determined by :attr:`output_file_name`. This method
        also ensures that the target directory exists before writing the file.

        :raises ValueError: If the :attr:`selected_profiles` instance variable
                            is None, indicating that no profiles have been selected
                            or labeled before attempting to write.
        """
        if self.selected_profiles is None:
            raise ValueError("Member variable 'selected_profiles' must not be empty.")

        os.makedirs(os.path.dirname(self.output_file_name), exist_ok=True)
        self.selected_profiles.write_parquet(self.output_file_name)
