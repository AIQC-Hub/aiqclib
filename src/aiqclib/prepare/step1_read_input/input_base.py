"""
This module defines the InputDataSetBase class, providing a foundational structure
for loading, preprocessing, and managing input data within the DMQC library.
It includes capabilities for reading various file formats, renaming columns,
and filtering rows based on configurable parameters, serving as a base for
domain-specific input data handling.
"""

from typing import Optional

import polars as pl

from aiqclib.common.base.config_base import ConfigBase
from aiqclib.common.base.dataset_base import DataSetBase
from aiqclib.common.utils.file import read_input_file


class InputDataSetBase(DataSetBase):
    """
    Base class for input data loading.

    It extends :class:`aiqclib.common.base.dataset_base.DataSetBase` by adding
    mechanisms for reading raw data from a file, renaming columns, and filtering rows.

    Subclasses must implement or customize methods such as :meth:`rename_columns`
    and :meth:`filter_rows` to handle domain-specific requirements.

    :ivar input_file_name: The absolute or resolved file path from which data will be read.
    :vartype input_file_name: str
    :ivar input_data: Polars DataFrame holding the loaded input data. Defaults to None
        until :meth:`read_input_data` is called.
    :vartype input_data: Optional[polars.DataFrame]
    """

    def __init__(self, config: ConfigBase) -> None:
        """
        Initialize the input dataset with a given configuration.

        :param config: The ConfigBase object providing path and parameter details.
        :type config: aiqclib.common.base.config_base.ConfigBase
        :raises NotImplementedError: If the ``expected_class_name`` is not defined
            by a subclass of :class:`aiqclib.common.base.dataset_base.DataSetBase`.
        :raises ValueError: If the YAML config does not match this class's
            expected class name.
        """
        super().__init__(step_name="input", config=config)

        self.input_file_name: str = self.config.get_full_file_name(
            step_name="input",
            default_file_name=self.config.data["input_file_name"],
            use_dataset_folder=False,
            folder_name_auto=False,
        )
        self.input_data: Optional[pl.DataFrame] = None

    def read_input_data(self) -> None:
        """
        Load data from the configured file into :attr:`input_data`.

        The method retrieves ``file_type`` and ``read_file_options`` from the config
        and uses :func:`aiqclib.common.utils.file.read_input_file` to read the file
        specified by :attr:`input_file_name`.

        After reading the data, it optionally calls :meth:`rename_columns` and
        :meth:`filter_rows` to modify the DataFrame.

        :raises FileNotFoundError: If the specified file cannot be found.
        :raises polars.exceptions.NoDataError: If the file is empty or cannot be parsed.
        :raises Exception: For other errors during file reading or processing.
        """
        input_file = self.input_file_name
        file_type = self.config.get_step_params("input").get("file_type")
        read_file_options = self.config.get_step_params("input").get(
            "read_file_options", {}
        )

        self.input_data = read_input_file(input_file, file_type, read_file_options)
        self.rename_columns()
        self.filter_rows()

    def rename_columns(self) -> None:
        """
        Rename columns in :attr:`input_data` using rename mappings from the config.

        If ``sub_steps.rename_columns`` is enabled and a ``rename_dict`` is present,
        columns will be renamed accordingly. Otherwise, the method does nothing.

        :raises polars.exceptions.ColumnNotFoundError: If a column specified in ``rename_dict``
            for renaming does not exist in the DataFrame.
        """
        if self.config.get_step_params("input")["sub_steps"][
            "rename_columns"
        ] and "rename_dict" in self.config.get_step_params("input"):
            self.input_data = self.input_data.rename(
                self.config.get_step_params("input")["rename_dict"]
            )

    def filter_rows(self) -> None:
        """
        Filter rows in :attr:`input_data` based on year constraints or other rules.

        If ``sub_steps.filter_rows`` is enabled and relevant fields exist,
        it will either remove certain years via :meth:`remove_years` or keep
        only a specified set of years via :meth:`keep_years`.

        :raises polars.exceptions.ColumnNotFoundError: If 'profile_timestamp' column is not
            present in :attr:`input_data` when year-based filtering is attempted.
        """
        input_params = self.config.get_step_params("input")
        if (
            not input_params["sub_steps"]["filter_rows"]
            or "filter_method_dict" not in input_params
        ):
            return
        filter_dict = input_params.get("filter_method_dict")

        if "remove_years" in filter_dict and len(filter_dict["remove_years"]) > 0:
            self.remove_years()

        if "keep_years" in filter_dict and len(filter_dict["keep_years"]) > 0:
            self.keep_years()

    def remove_years(self) -> None:
        """
        Remove data rows for years listed under ``remove_years`` in the config.

        Updates :attr:`input_data` by filtering out rows whose year is in
        the ``remove_years`` list. This method assumes the existence of a
        'profile_timestamp' column in :attr:`input_data` to extract the year.

        :raises polars.exceptions.ColumnNotFoundError: If 'profile_timestamp' column is not
            present in :attr:`input_data`.
        """
        years = self.config.get_step_params("input")["filter_method_dict"][
            "remove_years"
        ]
        self.input_data = (
            self.input_data.with_columns(
                pl.col("profile_timestamp").dt.year().alias("year")
            )
            .filter(~pl.col("year").is_in(years))
            .drop("year")
        )

    def keep_years(self) -> None:
        """
        Keep only data rows for years listed under ``keep_years`` in the config.

        Updates :attr:`input_data` by filtering in rows whose year is in
        the ``keep_years`` list. This method assumes the existence of a
        'profile_timestamp' column in :attr:`input_data` to extract the year.

        :raises polars.exceptions.ColumnNotFoundError: If 'profile_timestamp' column is not
            present in :attr:`input_data`.
        """
        years = self.config.get_step_params("input")["filter_method_dict"]["keep_years"]
        self.input_data = (
            self.input_data.with_columns(
                pl.col("profile_timestamp").dt.year().alias("year")
            )
            .filter(pl.col("year").is_in(years))
            .drop("year")
        )
