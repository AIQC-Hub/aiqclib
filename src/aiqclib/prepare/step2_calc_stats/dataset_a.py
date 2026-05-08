"""
This module defines the SummaryDataSetA class, a specialized implementation
of SummaryStatsBase for calculating summary statistics on specific datasets,
such as Copernicus CTD data, using the Polars DataFrame library.
It integrates with a configuration management system to ensure proper data processing.
"""

from typing import Optional

import polars as pl

from aiqclib.common.base.config_base import ConfigBase
from aiqclib.prepare.step2_calc_stats.summary_base import SummaryStatsBase


class SummaryDataSetA(SummaryStatsBase):
    """
    Specialized class for calculating summary statistics for Copernicus CTD data.

    This class extends :class:`aiqclib.prepare.step2_calc_stats.summary_base.SummaryStatsBase`
    and leverages the Polars DataFrame library for efficient data processing.
    It identifies itself via the :attr:`expected_class_name` attribute
    to match corresponding YAML configuration entries.
    """

    expected_class_name: str = "SummaryDataSetA"

    def __init__(
        self, config: ConfigBase, input_data: Optional[pl.DataFrame] = None
    ) -> None:
        """
        Initializes the SummaryDataSetA instance.

        This constructor sets up the summary statistics calculator with the provided
        configuration and an optional initial dataset.

        :param config: The configuration object containing all necessary parameters
                       and paths for data processing and summary statistics calculation.
        :type config: aiqclib.common.base.config_base.ConfigBase
        :param input_data: An optional Polars DataFrame to immediately process.
                           If ``None``, data should be set later using other methods.
        :type input_data: Optional[polars.DataFrame]
        """
        super().__init__(config=config, input_data=input_data)
