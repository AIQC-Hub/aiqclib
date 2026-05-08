"""
Module for locating specific data rows within oceanographic datasets.

This module defines the :class:`LocateDataSetA` class, which identifies positive
(bad quality) and negative (good quality) observations from oceanographic
profiles. It facilitates the creation of paired datasets for machine
learning by aligning observations based on profile and pressure proximity.
"""

from typing import Dict, Optional

import polars as pl

from aiqclib.common.base.config_base import ConfigBase
from aiqclib.prepare.step4_select_rows.locate_base import LocatePositionBase


class LocateDataSetA(LocatePositionBase):
    """
    A subclass of :class:`aiqclib.prepare.step4_select_rows.locate_base.LocatePositionBase`
    that locates both positive and negative rows from BO NRT+Cora test data for
    training or evaluation purposes.

    The workflow involves:

      - Selecting rows that have "bad" QC flags (positive examples).
      - Selecting rows that have "good" QC flags (negative examples).
      - Aligning these two sets to form paired data examples, often based on
        proximity in profile and pressure.
      - Concatenating and labeling them for subsequent steps in a machine
        learning pipeline.
    """

    expected_class_name: str = "LocateDataSetA"

    def __init__(
        self,
        config: ConfigBase,
        input_data: Optional[pl.DataFrame] = None,
        selected_profiles: Optional[pl.DataFrame] = None,
    ) -> None:
        """
        Initialize the dataset with configuration, an input DataFrame,
        and a DataFrame of selected profiles.

        :param config: A dataset configuration object specifying paths,
                       parameters, and target definitions for locating test data rows.
        :type config: aiqclib.common.base.config_base.ConfigBase
        :param input_data: A Polars DataFrame containing the full data
                           from which positive and negative rows will be derived.
                           Defaults to None.
        :type input_data: polars.DataFrame or None
        :param selected_profiles: A Polars DataFrame containing profiles
                                  that have already been labeled as positive or negative.
                                  Defaults to None.
        :type selected_profiles: polars.DataFrame or None
        """
        super().__init__(
            config=config, input_data=input_data, selected_profiles=selected_profiles
        )

        #: Dictionary for holding subsets of positive rows keyed by target name.
        self.positive_rows: Dict[str, pl.DataFrame] = {}
        #: Dictionary for holding subsets of negative rows keyed by target name.
        self.negative_rows: Dict[str, pl.DataFrame] = {}

    def select_positive_rows(self, target_name: str, target_value: Dict) -> None:
        """
        Identify and collect positive rows for a given target. Positive rows are
        defined as observations within profiles that have a specific "bad" QC flag.

        :param target_name: The name (key) of the target in the config's target dictionary.
        :type target_name: str
        :param target_value: A dictionary of target metadata, including the QC flag
                             variable name that indicates a "bad" observation (e.g., flag=4).
        :type target_value: Dict
        """
        pos_flag_values = target_value.get("pos_flag_values", [4])
        self.positive_rows[target_name] = (
            self.selected_profiles.filter(pl.col("label") == 1)
            .select(["profile_id", "pos_profile_id", "platform_code", "profile_no"])
            .join(
                (
                    self.input_data.filter(
                        pl.col(target_value["flag"]).is_in(pos_flag_values)
                    ).select(
                        pl.col("platform_code"),
                        pl.col("profile_no"),
                        pl.col("observation_no"),
                        pl.col("pres"),
                        pl.col(target_value["flag"]).alias("flag"),
                    )
                ),
                on=["platform_code", "profile_no"],
            )
            .with_columns(
                pl.concat_str(
                    ["platform_code", "profile_no", "observation_no"],
                    separator="|",
                ).alias("pair_id"),
                pl.lit(1).alias("label"),
            )
        )

    def select_negative_rows(self, target_name: str, target_value: Dict) -> None:
        """
        Identify and collect negative rows that align with positive rows,
        forming pairs where possible. Negative rows are typically "good"
        observations from nearby profiles, matched by pressure.

        :param target_name: The target name used to locate the corresponding positive rows.
        :type target_name: str
        :param target_value: A dictionary of target metadata, including the QC flag
                             variable name used for selecting negative observations
                             (e.g., flag=1 or any "good" flag).
        :type target_value: Dict
        """
        self.select_negative_rows_closest_day(target_name, target_value)
        neighbor_n = self.config.get_step_params("locate").get("neighbor_n", 0)
        if neighbor_n > 0:
            self.select_negative_rows_neighbor_n(target_name, target_value)

    def select_negative_rows_closest_day(
        self, target_name: str, target_value: Dict
    ) -> None:
        """
        Identify and collect negative rows that align with positive rows,
        forming pairs where possible. Negative rows are typically "good"
        observations from nearby profiles, matched by pressure.

        The alignment process involves:

        1. Selecting positive rows.
        2. Joining with negative profiles.
        3. Joining with the full input data to get observation details.
        4. Calculating pressure differences with corresponding positive observations.
        5. Selecting the negative observation that best matches in pressure
           for each positive observation to form a pair.

        :param target_name: The target name used to locate the corresponding positive rows.
        :type target_name: str
        :param target_value: A dictionary of target metadata, including the QC flag
                             variable name used for selecting negative observations
                             (e.g., flag=1 or any "good" flag).
        :type target_value: Dict
        """
        neg_flag_values = target_value.get("neg_flag_values", [1])
        negative_rows = (
            self.positive_rows[target_name]
            .select(
                pl.col("platform_code").alias("pos_platform_code"),
                pl.col("profile_no").alias("pos_profile_no"),
                pl.col("pos_profile_id"),
                pl.col("observation_no").alias("pos_observation_no"),
                pl.col("pres").alias("pos_pres"),
                pl.col("pair_id"),
            )
            .join(
                self.selected_profiles.filter(pl.col("label") == 0).select(
                    pl.col("profile_id"),
                    pl.col("pos_profile_id"),
                    pl.col("platform_code"),
                    pl.col("profile_no"),
                ),
                how="inner",
                on="pos_profile_id",
            )
            .join(
                self.input_data.filter(
                    pl.col(target_value["flag"]).is_in(neg_flag_values)
                ).select(
                    pl.col("platform_code"),
                    pl.col("profile_no"),
                    pl.col("observation_no"),
                    pl.col("pres"),
                    pl.col(target_value["flag"]).alias("flag"),
                ),
                how="inner",
                on=["platform_code", "profile_no"],
            )
            .with_columns(
                (pl.col("pos_pres") - pl.col("pres")).abs().alias("pres_diff")
            )
            .group_by(
                [
                    "pos_profile_id",
                    "pos_observation_no",
                    "profile_id",
                ]
            )
            .agg(pl.all().sort_by("pres_diff").first())
            .select(
                [
                    pl.col("profile_id"),
                    pl.col("platform_code"),
                    pl.col("profile_no"),
                    pl.col("observation_no"),
                    pl.col("pres"),
                    pl.col("flag"),
                    pl.col("pair_id"),
                    pl.lit(0).alias("label"),
                ]
            )
        )

        self.negative_rows[target_name] = negative_rows

    def select_negative_rows_neighbor_n(
        self, target_name: str, target_value: Dict
    ) -> None:
        """
        Identify and collect negative rows that align with positive rows,
        forming pairs where possible. Negative rows are typically "good"
        observations from nearby profiles, matched by pressure.

        The alignment process involves:

        1. Selecting positive profiles.
        2. Generating neighbouring observation numbers.
        3. Joining with the full input data to get observation details.
        4. Selecting the negative observations.

        :param target_name: The target name used to locate the corresponding positive rows.
        :type target_name: str
        :param target_value: A dictionary of target metadata, including the QC flag
                             variable name used for selecting negative observations
                             (e.g., flag=1 or any "good" flag).
        :type target_value: Dict
        """
        neighbor_n = self.config.get_step_params("locate").get("neighbor_n", 0)
        neighbor_no = list(range(1, neighbor_n + 1)) + [
            -x for x in range(1, neighbor_n + 1)
        ]
        neg_flag_values = target_value.get("neg_flag_values", [1])

        negative_rows = (
            self.positive_rows[target_name]
            .select(
                pl.col("profile_id"),
                pl.col("platform_code"),
                pl.col("profile_no"),
                pl.col("observation_no"),
                pl.col("pair_id"),
            )
            .join(
                pl.DataFrame({"neighbor_no": neighbor_no}),
                how="cross",
            )
            .with_columns(
                (pl.col("observation_no") + pl.col("neighbor_no"))
                .cast(pl.Int32)
                .alias("observation_no")
            )
            .join(
                self.input_data.filter(
                    pl.col(target_value["flag"]).is_in(neg_flag_values)
                ).select(
                    pl.col("platform_code"),
                    pl.col("profile_no"),
                    pl.col("observation_no"),
                    pl.col("pres"),
                    pl.col(target_value["flag"]).alias("flag"),
                ),
                how="inner",
                on=["platform_code", "profile_no", "observation_no"],
            )
            .select(
                [
                    pl.col("profile_id"),
                    pl.col("platform_code"),
                    pl.col("profile_no"),
                    pl.col("observation_no"),
                    pl.col("pres"),
                    pl.col("flag"),
                    pl.col("pair_id"),
                    pl.lit(0).alias("label"),
                ]
            )
        )
        self.negative_rows[target_name] = self.negative_rows[target_name].vstack(
            negative_rows
        )

    def locate_target_rows(self, target_name: str, target_value: Dict) -> None:
        """
        Locate training data rows by consolidating positive and negative subsets.
        This method first calls :meth:`select_positive_rows` and
        :meth:`select_negative_rows` to gather the respective dataframes,
        then stacks them, adds a unique row index, and creates a `pair_id`
        for linking paired observations.

        :param target_name: Name of the target variable (e.g., 'TEMP_QC').
        :type target_name: str
        :param target_value: A dictionary of target metadata, including the QC flag
                             variable name used for both positive and negative selection.
        :type target_value: Dict
        """
        self.select_positive_rows(target_name, target_value)
        self.select_negative_rows(target_name, target_value)

        self.selected_rows[target_name] = (
            self.positive_rows[target_name]
            .drop("pos_profile_id")
            .vstack(self.negative_rows[target_name])
            .with_row_index("row_id", offset=1)
        )
