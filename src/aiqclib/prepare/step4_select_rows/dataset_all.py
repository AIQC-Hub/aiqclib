"""
This module provides the LocateDataSetAll class, which is used for identifying and
extracting positive and negative data rows from oceanographic profiles. It filters
observations based on quality control (QC) flags to prepare datasets for machine
learning training or evaluation tasks.
"""

from typing import Dict, Optional

import polars as pl

from aiqclib.common.base.config_base import ConfigBase
from aiqclib.prepare.step4_select_rows.locate_base import LocatePositionBase


class LocateDataSetAll(LocatePositionBase):
    """
    A subclass of :class:`aiqclib.prepare.step4_select_rows.locate_base.LocatePositionBase`
    that locates both positive and negative rows from data for training or evaluation purposes.

    The workflow involves:
      - Selecting rows that have "bad" QC flags (positive examples).
      - Selecting rows that have "good" QC flags (negative examples).
      - Concatenating and labeling them for subsequent steps in a machine learning pipeline.
    """

    expected_class_name: str = "LocateDataSetAll"

    def __init__(
        self,
        config: ConfigBase,
        input_data: Optional[pl.DataFrame] = None,
        selected_profiles: Optional[pl.DataFrame] = None,
    ) -> None:
        """
        Initialize the dataset with configuration, an input DataFrame, and a
        DataFrame of selected profiles.

        :param config: A dataset configuration object specifying paths and parameters.
        :type config: aiqclib.common.base.config_base.ConfigBase
        :param input_data: A Polars DataFrame containing the full data to be processed.
        :type input_data: polars.DataFrame or None
        :param selected_profiles: A Polars DataFrame containing profiles that have already been labeled.
        :type selected_profiles: polars.DataFrame or None
        """
        super().__init__(
            config=config, input_data=input_data, selected_profiles=selected_profiles
        )

    def select_all_rows(self, target_name: str, target_value: Dict) -> None:
        """
        Collect all rows for a specified target by applying flag-based labeling to each record.

        :param target_name: The name (key) of the target in the configuration's target dictionary.
        :type target_name: str
        :param target_value: A dictionary of target metadata, including QC flag names and values.
        :type target_value: Dict
        :raises ValueError: If the internal input_data attribute is None.
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
        Locate target rows for training or evaluation by calling select_all_rows.

        :param target_name: Name of the target variable.
        :type target_name: str
        :param target_value: A dictionary of target metadata used for labeling.
        :type target_value: Dict
        """
        self.select_all_rows(target_name, target_value)
