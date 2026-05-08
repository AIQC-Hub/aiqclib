"""
This module defines the ExtractDataSetA class, a specialized feature extraction class
for Copernicus CTD data. It extends ExtractFeatureBase to implement
specific data processing and feature generation steps for this dataset,
integrating with the aiqclib framework's configuration and data flow.
"""

from typing import Optional, Dict

import polars as pl

from aiqclib.common.base.config_base import ConfigBase
from aiqclib.prepare.step5_extract_features.extract_base import ExtractFeatureBase


class ExtractDataSetA(ExtractFeatureBase):
    """
    A subclass of :class:`ExtractFeatureBase` designed to extract features
    specifically from Copernicus CTD data.

    This class sets its :attr:`expected_class_name` to ``"ExtractDataSetA"``,
    ensuring it is recognized in the YAML configuration as a valid
    extract class within the aiqclib framework. It inherits the full
    feature extraction pipeline and lifecycle management from its base class,
    :class:`ExtractFeatureBase`.

    :cvar expected_class_name: The name expected in configuration files to identify this class.
    :vartype expected_class_name: str
    """

    expected_class_name: str = "ExtractDataSetA"

    def __init__(
        self,
        config: ConfigBase,
        input_data: Optional[pl.DataFrame] = None,
        selected_profiles: Optional[pl.DataFrame] = None,
        selected_rows: Optional[Dict[str, pl.DataFrame]] = None,
        summary_stats: Optional[pl.DataFrame] = None,
    ) -> None:
        """
        Initializes the feature extraction workflow for Copernicus CTD data.

        This constructor sets up the necessary data and configuration for the
        feature extraction process, leveraging the capabilities of the base class.

        :param config: A dataset configuration object that manages paths, target definitions, and parameters.
        :type config: :class:`~aiqclib.common.base.config_base.ConfigBase`
        :param input_data: An optional Polars DataFrame containing the full pre-processed dataset.
        :type input_data: :class:`polars.DataFrame` or None
        :param selected_profiles: An optional Polars DataFrame containing specifically-selected profiles.
        :type selected_profiles: :class:`polars.DataFrame` or None
        :param selected_rows: An optional mapping of target names to their respective subset of rows.
        :type selected_rows: Dict[str, :class:`polars.DataFrame`] or None
        :param summary_stats: An optional Polars DataFrame with summary statistics for feature scaling.
        :type summary_stats: :class:`polars.DataFrame` or None
        """
        super().__init__(
            config=config,
            input_data=input_data,
            selected_profiles=selected_profiles,
            selected_rows=selected_rows,
            summary_stats=summary_stats,
        )
