"""
This module defines the SummaryDataSetAll class, a specialized class for
calculating summary statistics specifically for BO NRT (Near Real-Time) and
Cora test datasets using the Polars data manipulation library.

It extends SummaryStatsBase to provide a concrete implementation for these
specific data contexts, integrating with a ConfigBase object for path and
parameter management, and defining default output file names.
"""

import polars as pl
from typing import Optional

from aiqclib.common.base.config_base import ConfigBase
from aiqclib.prepare.step2_calc_stats.summary_base import SummaryStatsBase


class SummaryDataSetAll(SummaryStatsBase):
    """
    Subclass of SummaryStatsBase for calculating summary statistics
    for BO NRT and Cora test datasets (Copernicus CTD data) using Polars.

    Sets expected_class_name to 'SummaryDataSetAll' to match
    the relevant YAML configuration.

    :cvar expected_class_name: The name of the class used for configuration matching.
    :vartype expected_class_name: str
    """

    expected_class_name: str = "SummaryDataSetAll"

    def __init__(
        self, config: ConfigBase, input_data: Optional[pl.DataFrame] = None
    ) -> None:
        """
        Initialize SummaryDataSetAll with the provided configuration and optional data.

        :param config: Configuration object containing paths and parameters for
                       generating summary statistics.
        :type config: aiqclib.common.base.config_base.ConfigBase
        :param input_data: Optional Polars DataFrame that can be used to
                           calculate the summary statistics. If not provided,
                           it should be assigned later before calling
                           statistic-related methods.
        :type input_data: Optional[polars.DataFrame]
        """
        super().__init__(config=config, input_data=input_data)

        #: Default output file name for summary statistics; can be overridden if necessary.
        self.default_file_name: str = "summary_stats_classify.tsv"

        #: The resolved absolute path for writing the summary statistics file,
        #: based on the configuration and self.default_file_name.
        self.output_file_name: str = self.config.get_full_file_name(
            step_name="summary", default_file_name=self.default_file_name
        )
