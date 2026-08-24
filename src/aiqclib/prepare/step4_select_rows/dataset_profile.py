"""
This module provides the LocateDataSetProfile class, the profile-level
counterpart of :class:`LocateDataSetAll`. Instead of one labeled row per
observation, it produces one labeled row per profile, with a binary or
proportion label derived from the QC flags of the profile's observations.
"""

import os
from typing import Dict, Optional

import polars as pl

from aiqclib.common.base.config_base import ConfigBase
from aiqclib.common.constants import PROFILE_KEYS
from aiqclib.common.utils.qc_flags import flag_as_int, flag_is_in
from aiqclib.prepare.step4_select_rows.locate_base import LocatePositionBase


class LocateDataSetProfile(LocatePositionBase):
    """
    A subclass of :class:`aiqclib.prepare.step4_select_rows.locate_base.LocatePositionBase`
    that produces one labeled row per profile for the profile-level pipeline.

    Per target, the workflow is:

      - Restrict the input to the selected profiles and to observations whose
        QC flag carries a valid (positive or negative) value; keep these rows
        in :attr:`observation_rows` for per-profile feature aggregation.
      - Aggregate the valid-flagged observations per profile into a single
        labeled row in :attr:`selected_rows`, with the label controlled by the
        target's ``label_mode``:

        - ``binary`` (default): 1 when any observation is positive-flagged,
          else 0.
        - ``proportion``: the fraction of positive-flagged observations among
          the valid-flagged ones, a float in [0, 1].

      Profiles without any valid-flagged observation are dropped.

    The per-profile frames have no ``observation_no`` column; downstream
    steps treat the identity columns as a tolerant superset (see
    :mod:`aiqclib.common.constants`).
    """

    expected_class_name: str = "LocateDataSetProfile"

    def __init__(
        self,
        config: ConfigBase,
        input_data: Optional[pl.DataFrame] = None,
        selected_profiles: Optional[pl.DataFrame] = None,
    ) -> None:
        """
        Initialize the profile-level locate step.

        :param config: A dataset configuration object specifying paths and parameters.
        :type config: aiqclib.common.base.config_base.ConfigBase
        :param input_data: A Polars DataFrame containing the full data to be processed.
        :type input_data: polars.DataFrame or None
        :param selected_profiles: A Polars DataFrame of profiles selected in step 3.
        :type selected_profiles: polars.DataFrame or None
        """
        super().__init__(
            config=config, input_data=input_data, selected_profiles=selected_profiles
        )

        #: File name template for the per-target observation rows kept for
        #: feature aggregation.
        self.default_observation_file_name: str = (
            "selected_observation_rows_{target_name}.parquet"
        )

        #: Output file paths for :attr:`observation_rows`, one per target.
        self.observation_output_file_names: Dict[str, str] = (
            self.config.get_target_file_names(
                step_name="locate",
                default_file_name=self.default_observation_file_name,
            )
        )

        #: The valid-flagged observation rows per target (the
        #: :class:`LocateDataSetAll` schema), used by the profile-level
        #: extract step to aggregate observation-level features.
        self.observation_rows: Dict[str, pl.DataFrame] = {}

    def select_observation_rows(self, target_name: str, target_value: Dict) -> None:
        """
        Collect the valid-flagged observation rows for a target, restricted to
        the selected profiles, and store them in :attr:`observation_rows`.

        :param target_name: The name (key) of the target in the configuration's
                            target dictionary.
        :type target_name: str
        :param target_value: A dictionary of target metadata, including QC flag
                             names and values.
        :type target_value: Dict
        :raises ValueError: If :attr:`input_data` or :attr:`selected_profiles`
                            is None.
        """
        if self.input_data is None:
            raise ValueError("Member variable 'input_data' must not be empty.")
        if self.selected_profiles is None:
            raise ValueError("Member variable 'selected_profiles' must not be empty.")

        pos_flag_values = target_value.get("pos_flag_values", [4])
        neg_flag_values = target_value.get("neg_flag_values", [1])
        flag_var_name = target_value["flag"]

        self.observation_rows[target_name] = (
            self.input_data.join(
                self.selected_profiles.select(PROFILE_KEYS).unique(),
                on=PROFILE_KEYS,
            )
            .with_row_index("row_id", offset=1)
            .filter(flag_is_in(flag_var_name, pos_flag_values + neg_flag_values))
            .with_columns(
                pl.lit(0, dtype=pl.UInt32).alias("profile_id"),
                pl.lit("").alias("pair_id"),
                pl.when(flag_is_in(flag_var_name, pos_flag_values))
                .then(1)
                .when(flag_is_in(flag_var_name, neg_flag_values))
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
                flag_as_int(flag_var_name).alias("flag"),
                pl.col("label"),
                pl.col("pair_id"),
            )
        )

    def select_profile_rows(self, target_name: str) -> None:
        """
        Aggregate :attr:`observation_rows` into one labeled row per profile,
        stored in :attr:`selected_rows`.

        The label is controlled by the target's ``label_mode`` (see the class
        docstring). Profiles without valid-flagged observations never appear
        here because :attr:`observation_rows` only holds valid-flagged rows.

        :param target_name: The name of the target being processed.
        :type target_name: str
        """
        label_mode = self.config.get_label_mode(target_name)

        if label_mode == "proportion":
            label_expr = (
                (pl.col("label").sum() / pl.col("label").count())
                .cast(pl.Float64)
                .alias("label")
            )
        else:
            label_expr = (pl.col("label").sum() > 0).cast(pl.UInt32).alias("label")

        self.selected_rows[target_name] = (
            self.observation_rows[target_name]
            .group_by(PROFILE_KEYS)
            .agg(label_expr)
            .sort(PROFILE_KEYS)
            .with_row_index("row_id", offset=1)
            .with_columns(
                pl.lit(0, dtype=pl.UInt32).alias("profile_id"),
                pl.lit("").alias("pair_id"),
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
        Locate the profile-level rows for a target: first the valid-flagged
        observation rows, then their per-profile aggregation.

        :param target_name: Name of the target variable.
        :type target_name: str
        :param target_value: A dictionary of target metadata used for labeling.
        :type target_value: Dict
        """
        self.select_observation_rows(target_name, target_value)
        self.select_profile_rows(target_name)

    def write_selected_rows(self) -> None:
        """
        Write both the per-profile rows and the per-target observation rows.

        The per-profile frames go to the standard ``selected_rows_{target}``
        files; the observation rows go to
        ``selected_observation_rows_{target}`` files for debuggability and
        re-use.
        """
        super().write_selected_rows()

        for target_name, df in self.observation_rows.items():
            file_path = self.observation_output_file_names[target_name]
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            df.write_parquet(file_path)
