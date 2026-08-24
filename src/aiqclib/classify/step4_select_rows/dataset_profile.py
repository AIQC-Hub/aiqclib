"""
This module provides the classification-stage LocateDataSetProfile class,
the profile-level counterpart of the classify :class:`LocateDataSetAll`.

Per target it emits one row per profile. Targets with a QC flag get the same
binary/proportion label as in preparation (evaluation is possible); targets
without a flag (``skip_evaluation``) keep every profile with a null label,
so unlabeled data can be classified.
"""

from typing import Dict, Optional

import polars as pl

from aiqclib.common.base.config_base import ConfigBase
from aiqclib.common.constants import PROFILE_KEYS
from aiqclib.prepare.step4_select_rows.dataset_profile import (
    LocateDataSetProfile as PrepareLocateDataSetProfile,
)


class LocateDataSetProfile(PrepareLocateDataSetProfile):
    """
    Profile-level row selection for the classification pipeline.

    Reuses the preparation-stage logic (valid-flagged observation rows plus
    a per-profile label honouring ``label_mode``) and adds the label-free
    path: when a target has no usable QC flag
    (:meth:`~aiqclib.common.base.config_base.ConfigBase.get_skip_evaluation`),
    every profile is kept with a null label and every observation is kept in
    :attr:`observation_rows` for feature aggregation.
    """

    expected_class_name: str = "LocateDataSetProfile"

    def __init__(
        self,
        config: ConfigBase,
        input_data: Optional[pl.DataFrame] = None,
        selected_profiles: Optional[pl.DataFrame] = None,
    ) -> None:
        """
        Initialize the classification-stage profile-level locate step.

        :param config: A classification configuration object.
        :type config: aiqclib.common.base.config_base.ConfigBase
        :param input_data: A Polars DataFrame containing the full data to be processed.
        :type input_data: polars.DataFrame or None
        :param selected_profiles: A Polars DataFrame of profiles selected in step 3.
        :type selected_profiles: polars.DataFrame or None
        """
        super().__init__(
            config=config, input_data=input_data, selected_profiles=selected_profiles
        )

        #: Classification-stage file name templates (mirroring the classify
        #: LocateDataSetAll convention).
        self.default_file_name: str = "selected_rows_classify_{target_name}.parquet"
        self.output_file_names: Dict[str, str] = self.config.get_target_file_names(
            step_name="locate", default_file_name=self.default_file_name
        )
        self.default_observation_file_name = (
            "selected_observation_rows_classify_{target_name}.parquet"
        )
        self.observation_output_file_names = self.config.get_target_file_names(
            step_name="locate",
            default_file_name=self.default_observation_file_name,
        )

    def select_label_free_rows(self, target_name: str) -> None:
        """
        Keep every observation and every profile of the selected profiles for
        a label-free target, with null ``flag``/``label``.

        :param target_name: The name of the target being processed.
        :type target_name: str
        :raises ValueError: If :attr:`input_data` or :attr:`selected_profiles`
                            is None.
        """
        if self.input_data is None:
            raise ValueError("Member variable 'input_data' must not be empty.")
        if self.selected_profiles is None:
            raise ValueError("Member variable 'selected_profiles' must not be empty.")

        self.observation_rows[target_name] = (
            self.input_data.join(
                self.selected_profiles.select(PROFILE_KEYS).unique(),
                on=PROFILE_KEYS,
            )
            .with_row_index("row_id", offset=1)
            .with_columns(
                pl.lit(0, dtype=pl.UInt32).alias("profile_id"),
                pl.lit("").alias("pair_id"),
                pl.lit(None, dtype=pl.Int64).alias("flag"),
                pl.lit(None, dtype=pl.Int64).alias("label"),
            )
            .select(
                pl.col("row_id"),
                pl.col("profile_id"),
                pl.col("platform_code"),
                pl.col("profile_no"),
                pl.col("observation_no"),
                pl.col("pres"),
                pl.col("flag"),
                pl.col("label"),
                pl.col("pair_id"),
            )
        )

        self.selected_rows[target_name] = (
            self.observation_rows[target_name]
            .unique(subset=PROFILE_KEYS)
            .sort(PROFILE_KEYS)
            .with_row_index("row_id_profile", offset=1)
            .with_columns(
                pl.col("row_id_profile").alias("row_id"),
                pl.lit(None, dtype=pl.Int64).alias("label"),
            )
            .select(
                pl.col("row_id"),
                pl.col("profile_id"),
                pl.col("platform_code"),
                pl.col("profile_no"),
                pl.col("label"),
                pl.col("pair_id"),
            )
        )

    def locate_target_rows(self, target_name: str, target_value: Dict) -> None:
        """
        Locate profile-level rows: the label-free path keeps every profile,
        otherwise the labelled preparation logic is reused.

        :param target_name: Name of the target variable.
        :type target_name: str
        :param target_value: A dictionary of target metadata used for labeling.
        :type target_value: Dict
        """
        if self.config.get_skip_evaluation(target_name):
            self.select_label_free_rows(target_name)
            return

        super().locate_target_rows(target_name, target_value)
