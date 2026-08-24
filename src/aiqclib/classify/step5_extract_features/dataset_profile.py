"""
This module provides the classification-stage ExtractDataSetProfile class.

It reuses the preparation-stage profile-level extraction logic (direct
profile features plus per-profile aggregation of observation-level features)
with the classification file-naming convention and ``normalization_role``
set to ``"apply"``: normalization values are loaded from the file produced
during preparation rather than re-derived from data.
"""

from typing import Dict, Optional

import polars as pl

from aiqclib.common.base.config_base import ConfigBase
from aiqclib.prepare.step5_extract_features.dataset_profile import (
    ExtractDataSetProfile as PrepareExtractDataSetProfile,
)


class ExtractDataSetProfile(PrepareExtractDataSetProfile):
    """
    Profile-level feature extraction for the classification pipeline.

    Mirrors how the classify :class:`ExtractDataSetAll` specializes the
    prepare extract step: same extraction logic, classification output file
    names, and previously-fitted normalization applied instead of fitted.
    """

    expected_class_name: str = "ExtractDataSetProfile"

    #: At classification time, normalization values are loaded from the file
    #: produced during preparation rather than re-derived from data.
    normalization_role: str = "apply"

    def __init__(
        self,
        config: ConfigBase,
        input_data: Optional[pl.DataFrame] = None,
        selected_profiles: Optional[pl.DataFrame] = None,
        selected_rows: Optional[Dict[str, pl.DataFrame]] = None,
        summary_stats: Optional[pl.DataFrame] = None,
        observation_rows: Optional[Dict[str, pl.DataFrame]] = None,
    ) -> None:
        """
        Initialize the classification-stage profile-level extract step.

        :param config: A classification configuration object.
        :type config: ConfigBase
        :param input_data: Polars DataFrame containing the raw input data.
        :type input_data: Optional[pl.DataFrame]
        :param selected_profiles: DataFrame of selected profiles.
        :type selected_profiles: Optional[pl.DataFrame]
        :param selected_rows: A mapping of target names to one-row-per-profile frames.
        :type selected_rows: Optional[Dict[str, pl.DataFrame]]
        :param summary_stats: DataFrame containing statistics for normalization.
        :type summary_stats: Optional[pl.DataFrame]
        :param observation_rows: Per-target observation rows kept by the locate
                                 step for feature aggregation.
        :type observation_rows: Optional[Dict[str, pl.DataFrame]]
        """
        super().__init__(
            config=config,
            input_data=input_data,
            selected_profiles=selected_profiles,
            selected_rows=selected_rows,
            summary_stats=summary_stats,
            observation_rows=observation_rows,
        )

        #: Classification-stage file name template.
        self.default_file_name: str = (
            "extracted_features_classify_{target_name}.parquet"
        )
        self.output_file_names: Dict[str, str] = self.config.get_target_file_names(
            step_name="extract", default_file_name=self.default_file_name
        )

        #: Working columns dropped from the final feature set.
        self.drop_col_names = [
            "profile_id",
            "pair_id",
        ]
