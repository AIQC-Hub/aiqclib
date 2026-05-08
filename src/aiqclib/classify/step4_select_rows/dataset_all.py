"""
Module for selecting all data rows from combined Copernicus CTD data.

This module provides the :class:`LocateDataSetAll` class, which extends
:class:`LocatePositionBase` to identify and label data points for machine
learning tasks based on Quality Control (QC) flags.

"""

from typing import Dict, Optional

import polars as pl

from aiqclib.common.base.config_base import ConfigBase
from aiqclib.prepare.step4_select_rows.locate_base import LocatePositionBase


class LocateDataSetAll(LocatePositionBase):
    """
    A subclass of :class:`LocatePositionBase` that locates all rows
    from Copernicus CTD data for training or evaluation purposes.

    This class assigns a default file naming scheme for target rows
    and uses configuration details (e.g., QC flags) to identify
    relevant data rows for each target.

    :cvar expected_class_name: The expected name of the class for validation.
    :vartype expected_class_name: str
    """

    expected_class_name: str = "LocateDataSetAll"

    def __init__(
        self,
        config: ConfigBase,
        input_data: Optional[pl.DataFrame] = None,
        selected_profiles: Optional[pl.DataFrame] = None,
    ) -> None:
        """
        Initialize the dataset with configuration, an optional input DataFrame,
        and an optional DataFrame of selected profiles.

        :param config: A configuration object specifying paths, parameters,
                       and target definitions for locating test data rows.
        :type config: aiqclib.common.base.config_base.ConfigBase
        :param input_data: An optional Polars DataFrame containing the full data
                           from which positive and negative rows will be derived.
                           If not provided, it should be set later using
                           :meth:`set_input_data`.
        :type input_data: polars.DataFrame or None
        :param selected_profiles: An optional Polars DataFrame containing profiles
                                  labeled as positive or negative. If not provided,
                                  it should be set later using
                                  :meth:`set_selected_profiles`.
        :type selected_profiles: polars.DataFrame or None
        """
        super().__init__(
            config=config, input_data=input_data, selected_profiles=selected_profiles
        )

        #: Default file name template for writing target rows (one file per target).
        self.default_file_name: str = "selected_rows_classify_{target_name}.parquet"

        #: Dictionary mapping each target name to the corresponding output Parquet file path.
        self.output_file_names: Dict[str, str] = self.config.get_target_file_names(
            step_name="locate", default_file_name=self.default_file_name
        )

    def select_all_rows(self, target_name: str, target_value: Dict) -> None:
        """
        Collect all rows for a specified target by applying
        flag-based labeling to each record.

        This method assumes that :attr:`input_data` has been set prior to its call.

        :param target_name: The name (key) of the target in the
                            configuration's target dictionary.
        :type target_name: str
        :param target_value: A dictionary of target metadata,
                             including the relevant QC flag variable name,
                             positive flag values, and negative flag values.
        :type target_value: Dict
        :raises ValueError: If :attr:`input_data` is None when this method is called.
        :raises KeyError: If 'flag' is not present in target_value.
        """
        if self.input_data is None:
            raise ValueError("Member variable 'input_data' must not be empty.")

        pos_flag_values = target_value.get("pos_flag_values", [4])
        neg_flag_values = target_value.get("neg_flag_values", [1])
        flag_var_name = target_value["flag"]
        self.selected_rows[target_name] = (
            self.input_data.with_row_index("row_id", offset=1)
            .filter(pl.col(flag_var_name).is_in(pos_flag_values + neg_flag_values))
            .with_columns(
                pl.lit(0, dtype=pl.UInt32).alias("profile_id"),
                pl.lit("").alias("pair_id"),
                pl.when(pl.col(flag_var_name).is_in(pos_flag_values))
                .then(1)
                .when(pl.col(flag_var_name).is_in(neg_flag_values))
                .then(0)
                .otherwise(None)
                .alias("label"),
            )
            .select(
                pl.col("row_id"),
                pl.col("profile_id"),
                pl.col("platform_code"),
                pl.col("profile_no"),
                pl.col("observation_no"),
                pl.col("pres"),
                pl.col(flag_var_name).alias("flag"),
                pl.col("label"),
                pl.col("pair_id"),
            )
        )

    def locate_target_rows(self, target_name: str, target_value: Dict) -> None:
        """
        Locate target rows for training or evaluation by calling :meth:`select_all_rows`.

        This method acts as a wrapper, ensuring all rows are considered for the target
        based on the provided QC flag.

        :param target_name: Name of the target variable.
        :type target_name: str
        :param target_value: A dictionary of target metadata, including
                             the QC flag variable name used for labeling
                             (e.g., ``{"flag": "TEMP_QC_FLAG"}``).
        :type target_value: Dict
        """
        self.select_all_rows(target_name, target_value)
