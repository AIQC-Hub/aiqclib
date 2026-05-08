"""
This module provides the ExtractDataSetAll class, which is designed for extracting
features from Copernicus CTD (Conductivity, Temperature, and Depth) datasets.
It inherits from ExtractFeatureBase and utilizes a configuration-driven approach
to define data targets and output paths.
"""

from typing import Optional, Dict

import polars as pl

from aiqclib.common.base.config_base import ConfigBase
from aiqclib.prepare.step5_extract_features.extract_base import ExtractFeatureBase


class ExtractDataSetAll(ExtractFeatureBase):
    """
    Feature extraction implementation specifically for Copernicus CTD data.

    This class serves as a concrete implementation of the :class:`ExtractFeatureBase`
    interface, specializing in the configuration and file naming conventions
    required for full CTD dataset processing.

    :cvar expected_class_name: The identifier used to match this class with
                               configuration settings.
    :vartype expected_class_name: str
    """

    expected_class_name: str = "ExtractDataSetAll"

    def __init__(
        self,
        config: ConfigBase,
        input_data: Optional[pl.DataFrame] = None,
        selected_profiles: Optional[pl.DataFrame] = None,
        selected_rows: Optional[Dict[str, pl.DataFrame]] = None,
        summary_stats: Optional[pl.DataFrame] = None,
    ) -> None:
        """
        Initialize the ExtractDataSetAll class with configuration and optional data.

        :param config: The configuration instance providing parameters and paths.
        :type config: ConfigBase
        :param input_data: Polars DataFrame containing the raw input data.
        :type input_data: Optional[pl.DataFrame]
        :param selected_profiles: DataFrame containing metadata or IDs for selected profiles.
        :type selected_profiles: Optional[pl.DataFrame]
        :param selected_rows: A mapping of target names to DataFrames containing specific row data.
        :type selected_rows: Optional[Dict[str, pl.DataFrame]]
        :param summary_stats: DataFrame containing statistics for normalization or scaling.
        :type summary_stats: Optional[pl.DataFrame]
        """
        super().__init__(
            config=config,
            input_data=input_data,
            selected_profiles=selected_profiles,
            selected_rows=selected_rows,
            summary_stats=summary_stats,
        )

        #: Default string template for naming exported feature files.
        self.default_file_name: str = (
            "extracted_features_classify_{target_name}.parquet"
        )

        #: Resolved mapping of target names to their specific output file paths.
        self.output_file_names: Dict[str, str] = self.config.get_target_file_names(
            step_name="extract", default_file_name=self.default_file_name
        )

        #: List of columns to be excluded or dropped during the extraction process.
        self.drop_col_names = [
            "profile_id",
            "pair_id",
        ]
