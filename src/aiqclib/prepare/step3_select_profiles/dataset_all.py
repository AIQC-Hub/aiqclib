"""
Module for selecting and labeling oceanographic profiles based on QC flags.

This module defines the :class:`SelectDataSetAll` class, which identifies
"bad" (positive) and "good" (negative) profiles based on Quality Control (QC)
criteria and prepares a labeled dataset for machine learning applications.
"""

import operator
from functools import reduce
from typing import Optional, List

import polars as pl

from aiqclib.common.base.config_base import ConfigBase
from aiqclib.prepare.step3_select_profiles.select_base import ProfileSelectionBase


class SelectDataSetAll(ProfileSelectionBase):
    """
    Selects positive/negative profiles from Copernicus CTD data.

    This class implements a strategy for labeling oceanographic profiles as
    "positive" (bad) or "negative" (good) based on their quality control (QC)
    flags.

    :ivar expected_class_name: The expected name of the class for config validation.
    :vartype expected_class_name: str
    :ivar pos_profile_df: DataFrame containing positively-labeled profiles.
    :vartype pos_profile_df: Optional[polars.DataFrame]
    :ivar neg_profile_df: DataFrame containing negatively-labeled profiles.
    :vartype neg_profile_df: Optional[polars.DataFrame]
    :ivar key_col_names: Column names used as unique identifiers for profiles.
    :vartype key_col_names: List[str]
    """

    expected_class_name: str = "SelectDataSetAll"

    def __init__(
        self, config: ConfigBase, input_data: Optional[pl.DataFrame] = None
    ) -> None:
        """
        Initialize the selection and labeling process.

        :param config: The configuration object containing paths and QC flag definitions.
        :type config: aiqclib.common.base.config_base.ConfigBase
        :param input_data: A Polars DataFrame containing the full set of profiles.
        :type input_data: Optional[polars.DataFrame]
        """
        super().__init__(config=config, input_data=input_data)

        self.pos_profile_df: Optional[pl.DataFrame] = None
        self.neg_profile_df: Optional[pl.DataFrame] = None
        self.key_col_names: List[str] = [
            "platform_code",
            "profile_no",
            "profile_timestamp",
            "longitude",
            "latitude",
        ]

    def select_positive_profiles(self) -> None:
        """
        Select profiles with "bad" QC flags.

        A profile is considered "positive" if any of its measurements have a QC
        flag defined as a positive flag in the configuration. Results are
        stored in :attr:`pos_profile_df`.
        """
        conditions = reduce(
            operator.or_,
            [
                pl.col(param["flag"]).is_in(param.get("pos_flag_values", [4]))
                for param in self.config.get_target_dict().values()
            ],
        )

        self.pos_profile_df = (
            self.input_data.filter(conditions)
            .select(self.key_col_names)
            .unique()
            .sort(["platform_code", "profile_no"])
            .with_row_index("profile_id", offset=1)
            .with_columns(
                pl.lit(0, dtype=pl.UInt32).alias("neg_profile_id"),
                pl.lit(1, dtype=pl.UInt32).alias("label"),
            )
        )

    def select_negative_profiles(self) -> None:
        """
        Select profiles with consistently "good" QC flags.

        A profile is considered "negative" if no measurements have a "bad" flag
        and at least one measurement has a "good" flag for all monitored parameters.
        Results are stored in :attr:`neg_profile_df`.
        """
        exprs = reduce(
            operator.and_,
            [
                (~pl.col(param["flag"]).is_in(param.get("pos_flag_values", [4])).any())
                & (pl.col(param["flag"]).is_in(param.get("neg_flag_values", [1])).any())
                for param in self.config.get_target_dict().values()
            ],
        )

        self.neg_profile_df = (
            self.input_data.filter(exprs.over(self.key_col_names))
            .select(self.key_col_names)
            .unique()
            .sort(["platform_code", "profile_no"])
            .with_row_index("profile_id", offset=self.pos_profile_df.shape[0] + 1)
            .with_columns(
                pl.lit(0, dtype=pl.UInt32).alias("neg_profile_id"),
                pl.lit(0, dtype=pl.UInt32).alias("label"),
            )
        )

    def label_profiles(self) -> None:
        """
        Execute the full profile selection and labeling workflow.

        Orchestrates the identification of positive and negative profiles and
        vstacks them into the :attr:`selected_profiles` attribute.
        """
        self.select_positive_profiles()
        self.select_negative_profiles()

        self.selected_profiles = self.pos_profile_df.vstack(self.neg_profile_df)
