"""
Summary Statistics Module.

This module provides the :class:`SummaryStatsBase` class, which serves as a base
for calculating, aggregating, and exporting summary statistics from tabular
datasets using the Polars library. It handles global and per-profile calculations
and supports exporting results to TSV format.
"""

import os
from typing import Optional, List

import polars as pl

from aiqclib.common.base.config_base import ConfigBase
from aiqclib.common.base.dataset_base import DataSetBase


class SummaryStatsBase(DataSetBase):
    """Abstract base class for calculating summary statistics.

    This class provides a framework for generating and writing summary
    statistics for a dataset. It handles both global (dataset-wide) and
    per-profile statistics for a specified set of numeric columns. Subclasses
    must define an ``expected_class_name`` to be instantiated.

    :ivar default_file_name: The default filename for the output stats file.
    :vartype default_file_name: str
    :ivar output_file_name: The full path for the output summary stats file,
                            derived from the configuration.
    :vartype output_file_name: str
    :ivar input_data: The DataFrame containing the data to be analyzed.
    :vartype input_data: polars.DataFrame or None
    :ivar summary_stats: DataFrame holding the combined global and per-profile
                         statistics after calculation.
    :vartype summary_stats: polars.DataFrame or None
    :ivar summary_stats_observation: DataFrame holding aggregated global statistics
                                     for key variables.
    :vartype summary_stats_observation: polars.DataFrame or None
    :ivar summary_stats_profile: DataFrame holding aggregated per-profile statistics
                                 for key variables.
    :vartype summary_stats_profile: polars.DataFrame or None
    :ivar val_col_names: List of numeric columns for which to compute statistics.
    :vartype val_col_names: list[str]
    :ivar stats_col_names: The schema (column names) for the output statistics
                           DataFrame.
    :vartype stats_col_names: list[str]
    :ivar profile_col_names: List of columns used to identify unique profiles for
                             grouping.
    :vartype profile_col_names: list[str]
    """

    def __init__(
        self, config: ConfigBase, input_data: Optional[pl.DataFrame] = None
    ) -> None:
        """Initialize the summary statistics calculation process.

        :param config: Configuration object that includes paths and parameters
                       for the statistics generation.
        :type config: aiqclib.common.base.config_base.ConfigBase
        :param input_data: A Polars DataFrame containing the data upon which
                           statistics will be computed. If None, it is
                           expected to be loaded by the base class.
        :type input_data: polars.DataFrame or None
        :raises NotImplementedError: If ``expected_class_name`` is not defined by
                                     a subclass upon instantiation.
        :raises ValueError: If the configuration's ``base_class`` does not match
                            the ``expected_class_name``.
        """
        super().__init__(step_name="summary", config=config)

        self.default_file_name: str = "summary_stats.tsv"
        self.output_file_name: str = self.config.get_full_file_name(
            step_name="summary", default_file_name=self.default_file_name
        )
        self.input_data: Optional[pl.DataFrame] = input_data
        self.summary_stats: Optional[pl.DataFrame] = None
        self.summary_stats_observation: Optional[pl.DataFrame] = None
        self.summary_stats_profile: Optional[pl.DataFrame] = None

        self.val_col_names = [
            "longitude",
            "latitude",
        ] + list(self.config.get_target_dict().keys())
        self.stats_col_names = [
            "platform_code",
            "profile_no",
            "variable",
            "min",
            "pct2.5",
            "pct25",
            "mean",
            "median",
            "pct75",
            "pct97.5",
            "max",
            "sd",
        ]
        self.profile_col_names = ["platform_code", "profile_no"]

    @staticmethod
    def get_stats_expression(val_col_name: str) -> List[pl.Expr]:
        """Build a list of Polars expressions to compute summary statistics.

        :param val_col_name: The name of the column to analyze.
        :type val_col_name: str
        :returns: A list of Polars expressions for calculating min, max, mean,
                  median, quantiles, and standard deviation.
        :rtype: list[polars.Expr]
        """
        return [
            pl.col(val_col_name).min().cast(pl.Float64).alias("min"),
            pl.col(val_col_name).max().cast(pl.Float64).alias("max"),
            pl.col(val_col_name).mean().cast(pl.Float64).alias("mean"),
            pl.col(val_col_name).median().cast(pl.Float64).alias("median"),
            pl.col(val_col_name).quantile(0.25).cast(pl.Float64).alias("pct25"),
            pl.col(val_col_name).quantile(0.75).cast(pl.Float64).alias("pct75"),
            pl.col(val_col_name).quantile(0.025).cast(pl.Float64).alias("pct2.5"),
            pl.col(val_col_name).quantile(0.975).cast(pl.Float64).alias("pct97.5"),
            pl.col(val_col_name).std().cast(pl.Float64).alias("sd"),
        ]

    def calculate_global_stats(self, val_col_name: str) -> pl.DataFrame:
        """Compute global summary statistics for a specified column.

        These statistics are calculated across the entire dataset.

        :param val_col_name: Name of the column for which to calculate global
                             statistics.
        :type val_col_name: str
        :returns: A DataFrame with one row containing the summary statistics,
                  structured to be compatible with per-profile stats.
        :rtype: polars.DataFrame
        """
        return (
            self.input_data.select(self.get_stats_expression(val_col_name))
            .with_columns(
                pl.lit("all").alias("platform_code"),
                pl.lit(0).alias("profile_no"),
                pl.lit(val_col_name).alias("variable"),
            )
            .select(self.stats_col_names)
        )

    def calculate_profile_stats(
        self, grouped_df: pl.DataFrame, val_col_name: str
    ) -> pl.DataFrame:
        """Compute per-profile summary statistics for a column.

        :param grouped_df: A Polars DataFrame already grouped by profile
                           identifier columns (e.g., platform_code, profile_no).
        :type grouped_df: polars.DataFrame
        :param val_col_name: The name of the column for which to calculate
                             per-profile stats.
        :type val_col_name: str
        :returns: A DataFrame containing statistics for each profile.
        :rtype: polars.DataFrame
        """
        return (
            grouped_df.agg(self.get_stats_expression(val_col_name))
            .with_columns(pl.lit(val_col_name).alias("variable"))
            .select(self.stats_col_names)
        )

    def calculate_stats(self) -> None:
        """Calculate and combine global and per-profile statistics.

        This method computes statistics for each column in :attr:`val_col_names`
        at both the global and per-profile level, then concatenates them into
        a single DataFrame stored in :attr:`summary_stats`.

        :returns: None
        :rtype: None
        """
        global_stats = pl.concat(
            [self.calculate_global_stats(x) for x in self.val_col_names]
        )
        grouped_df = self.input_data.group_by(self.profile_col_names)
        profile_stats = pl.concat(
            [self.calculate_profile_stats(grouped_df, x) for x in self.val_col_names]
        )

        self.summary_stats = global_stats.vstack(profile_stats)

    def write_summary_stats(self) -> None:
        """Write the computed summary statistics to a TSV file.

        The output path is determined by :attr:`output_file_name`.

        :raises ValueError: If :attr:`summary_stats` has not been calculated yet.
        :returns: None
        :rtype: None
        """
        if self.summary_stats is None:
            raise ValueError("Member variable 'summary_stats' must not be empty.")

        os.makedirs(os.path.dirname(self.output_file_name), exist_ok=True)
        self.summary_stats.write_csv(self.output_file_name, separator="\t")

    def create_summary_stats_observation(self) -> None:
        """Create a summarized view of global observation statistics.

        This method filters the main statistics table for global ("all") data,
        selects a subset of key metrics, and stores the result in
        :attr:`summary_stats_observation`.

        :raises ValueError: If :attr:`summary_stats` has not been calculated yet.
        :returns: None
        :rtype: None
        """
        if self.summary_stats is None:
            raise ValueError("Member variable 'summary_stats' must not be empty.")

        self.summary_stats_observation = (
            self.summary_stats.filter(pl.col("platform_code") == "all")
            .drop(["platform_code", "profile_no"])
            .select(["variable", "min", "mean", "pct97.5", "max"])
            .sort(["variable"])
        )

    def create_summary_stats_profile(self) -> None:
        """Create a summarized view of per-profile statistics.

        This method filters the main statistics table for per-profile data,
        reshapes it to aggregate statistics (min, mean, max, etc.) across all
        profiles, and stores the result in :attr:`summary_stats_profile`.

        :raises ValueError: If :attr:`summary_stats` has not been calculated yet.
        :returns: None
        :rtype: None
        """
        if self.summary_stats is None:
            raise ValueError("Member variable 'summary_stats' must not be empty.")

        self.summary_stats_profile = (
            self.summary_stats.filter(
                (pl.col("platform_code") != "all")
                & ~pl.col("variable").is_in(["longitude", "latitude"])
            )
            .unpivot(
                index=["platform_code", "profile_no", "variable"], variable_name="stats"
            )
            .group_by(["variable", "stats"])
            .agg(
                min=pl.col("value").min(),
                mean=pl.col("value").mean(),
                pct97_5=pl.col("value").quantile(0.975),
                max=pl.col("value").max(),
            )
            .rename({"pct97_5": "pct97.5"})
            .sort(["variable", "stats"])
        )
