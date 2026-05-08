"""
This module defines the SelectDataSetA class for oceanographic profile selection.

The module provides functionality to categorize profiles as "positive" (containing
errors) or "negative" (clean) based on Quality Control (QC) flags and pairs
them based on temporal proximity for machine learning dataset construction.
"""

import operator
from functools import reduce
from typing import Optional, List

import polars as pl

from aiqclib.common.base.config_base import ConfigBase
from aiqclib.prepare.step3_select_profiles.select_base import ProfileSelectionBase


class SelectDataSetA(ProfileSelectionBase):
    """Selects positive/negative profiles from Copernicus CTD data.

    This class implements a strategy for labeling oceanographic profiles as
    "positive" (bad) or "negative" (good) based on their quality control (QC)
    flags. The main steps are:

    1.  **Select Positive Profiles**: Identify profiles with at least one "bad"
        QC flag (e.g., a value of 4) in key sensor measurements.
    2.  **Select Negative Profiles**: Identify profiles where all measurements for
        all key sensors are "good" (e.g., a QC flag of 1).
    3.  **Find Profile Pairs**: For each positive profile, find the temporally
        closest negative profile to create a balanced and relevant dataset.
    4.  **Combine Data**: Merge the labeled positive and negative profiles into a
        single DataFrame.

    :ivar expected_class_name: The expected name of the class, used for
        configuration validation.
    :vartype expected_class_name: str
    :ivar pos_profile_df: DataFrame containing positively-labeled profiles.
    :vartype pos_profile_df: Optional[polars.DataFrame]
    :ivar neg_profile_df: DataFrame containing negatively-labeled profiles.
    :vartype neg_profile_df: Optional[polars.DataFrame]
    :ivar key_col_names: Column names used as unique identifiers for profiles.
    :vartype key_col_names: List[str]
    """

    expected_class_name: str = "SelectDataSetA"

    def __init__(
        self, config: ConfigBase, input_data: Optional[pl.DataFrame] = None
    ) -> None:
        """Initialize the selection and labeling process.

        :param config: The configuration object containing paths, parameters,
                       and QC flag definitions for the selection process.
        :type config: aiqclib.common.base.config_base.ConfigBase
        :param input_data: A Polars DataFrame containing the full set
                           of profiles from which to select examples. If None,
                           it is expected to be loaded by the base class.
        :type input_data: Optional[polars.DataFrame]
        :return: None
        :rtype: None
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
        """Select profiles with "bad" QC flags.

        A profile is considered "positive" (i.e., contains errors) if any of
        its measurements have a QC flag defined as a positive flag in the
        configuration (e.g., a flag of 4). The resulting unique profiles
        are stored in the :attr:`pos_profile_df` attribute.

        :return: None
        :rtype: None
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
                pl.col("profile_timestamp").dt.ordinal_day().alias("pos_day_of_year")
            )
        )

    def select_negative_profiles(self) -> None:
        """Select profiles with consistently "good" QC flags.

        A profile is considered "negative" (i.e., contains only good data)
        if, for every monitored parameter (e.g., temperature, salinity),
        none of its measurements have a "bad" flag and at least one has a "good"
        flag. The resulting unique profiles are stored in the
        :attr:`neg_profile_df` attribute.

        :return: None
        :rtype: None
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
                pl.col("profile_timestamp").dt.ordinal_day().alias("neg_day_of_year")
            )
        )

    def find_profile_pairs(self) -> None:
        """Pair positive profiles with their temporally closest negative profile.

        This method reduces the set of negative profiles to only those that
        are the nearest in time to a positive profile. This helps create a
        more balanced and comparable dataset for training or analysis.

        This method updates :attr:`pos_profile_df` by adding ``label`` and
        ``neg_profile_id`` columns. It also updates :attr:`neg_profile_df`
        by filtering it to the matched profiles and adding corresponding labels.

        :return: None
        :rtype: None
        """
        neg_pos_ratio = self.config.get_step_params("select").get("neg_pos_ratio", 1)

        closest_neg_id = (
            self.pos_profile_df.join(self.neg_profile_df, how="cross", suffix="_neg")
            .with_columns(
                (pl.col("pos_day_of_year") - pl.col("neg_day_of_year"))
                .abs()
                .alias("day_diff")
            )
            .rename({"profile_id": "pos_profile_id"})
            .sort(["pos_profile_id", "day_diff", "profile_id_neg"])
            .group_by("pos_profile_id")
            .agg(pl.col("profile_id_neg").head(neg_pos_ratio).alias("profile_id"))
            .explode("profile_id")
        )

        self.pos_profile_df = self.pos_profile_df.with_columns(
            pl.col("profile_id").alias("pos_profile_id"), pl.lit(1).alias("label")
        ).drop("pos_day_of_year")

        self.neg_profile_df = (
            self.neg_profile_df.join(
                closest_neg_id,
                on="profile_id",
                how="inner",
            )
            .with_columns(
                pl.lit(0).alias("label"),
            )
            .drop("neg_day_of_year")
        )

    def label_profiles(self) -> None:
        """Execute the full profile selection and labeling workflow.

        This method orchestrates the process by calling, in order:

        1. :meth:`select_positive_profiles`
        2. :meth:`select_negative_profiles`
        3. :meth:`find_profile_pairs`

        The final combined DataFrame of labeled profiles is stored in the
        :attr:`selected_profiles` attribute of the base class.

        :return: None
        :rtype: None
        """
        self.select_positive_profiles()
        self.select_negative_profiles()
        self.find_profile_pairs()

        self.selected_profiles = self.pos_profile_df.vstack(self.neg_profile_df)
