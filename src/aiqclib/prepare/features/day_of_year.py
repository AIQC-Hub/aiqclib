"""
This module defines a feature extraction class, DayOfYearFeat,
that calculates the day of the year from timestamps.

It is designed to be part of a larger feature engineering pipeline,
extending the FeatureBase class to derive temporal features,
specifically the day-of-year, and optionally apply a sinusoidal
transformation for cyclical encoding.
"""

from typing import Optional, Dict

import numpy as np
import polars as pl

from aiqclib.common.base.feature_base import FeatureBase


class DayOfYearFeat(FeatureBase):
    """
    A feature-extraction class that derives day-of-year features
    from Copernicus CTD data.

    This class specifically leverages the ``profile_timestamp``
    column to generate a day-of-year value, optionally applying
    a sinusoidal transformation for cyclical encoding.
    """

    def __init__(
        self,
        target_name: Optional[str] = None,
        feature_info: Optional[Dict] = None,
        selected_profiles: Optional[pl.DataFrame] = None,
        filtered_input: Optional[pl.DataFrame] = None,
        selected_rows: Optional[Dict[str, pl.DataFrame]] = None,
        summary_stats: Optional[pl.DataFrame] = None,
    ) -> None:
        """
        Initialize the day-of-year feature extraction process.

        :param target_name: The name of the target variable used to index
                            :attr:`selected_rows`. Defaults to None.
        :type target_name: Optional[str]
        :param feature_info: A dictionary describing feature parameters,
                             which may include a "convert" key (e.g., "sine")
                             for sinusoidal transformations. Defaults to None.
        :type feature_info: Optional[Dict]
        :param selected_profiles: A Polars DataFrame containing a subset
                                  of profiles relevant to feature extraction.
                                  Defaults to None.
        :type selected_profiles: Optional[pl.DataFrame]
        :param filtered_input: (Unused in this feature class) A filtered Polars
                               DataFrame of input data for advanced merging.
                               Defaults to None.
        :type filtered_input: Optional[pl.DataFrame]
        :param selected_rows: A dictionary of target-specific DataFrames, each
                            containing rows relevant to that target. Defaults to None.
        :type selected_rows: Optional[Dict[str, pl.DataFrame]]
        :param summary_stats: (Unused in this feature class) A Polars DataFrame
                              containing statistical information for potential scaling.
                              Defaults to None.
        :type summary_stats: Optional[pl.DataFrame]
        :return: None
        :rtype: None
        """
        super().__init__(
            target_name=target_name,
            feature_info=feature_info,
            selected_profiles=selected_profiles,
            filtered_input=filtered_input,
            selected_rows=selected_rows,
            summary_stats=summary_stats,
        )

    def extract_features(self) -> None:
        """
        Derive the day-of-year feature from the ``profile_timestamp`` column
        in :attr:`selected_profiles` and merge it with the target rows.

        Steps:

          1. Select columns ``row_id``, ``platform_code``, and ``profile_no``
             from :attr:`selected_rows[target_name]`.
          2. Join the subset with ``profile_timestamp`` from :attr:`selected_profiles`
             based on ``platform_code`` and ``profile_no``.
          3. Compute the day of year from ``profile_timestamp`` via
             Polars' :func:`polars.Expr.dt.ordinal_day`.
          4. Remove columns no longer needed (i.e., the join keys and timestamp).

        :return: None
        :rtype: None
        """
        self.features = (
            self.selected_rows[self.target_name]
            .select(["row_id", "platform_code", "profile_no"])
            .join(
                self.selected_profiles.select(
                    ["platform_code", "profile_no", "profile_timestamp"]
                ).unique(),
                on=["platform_code", "profile_no"],
                maintain_order="left",
            )
            .with_columns(
                pl.col("profile_timestamp").dt.ordinal_day().alias("day_of_year"),
            )
            .drop(["platform_code", "profile_no", "profile_timestamp"])
        )

    def scale_first(self) -> None:
        """
        (Optional) Perform the initial scaling step.

        Currently, no transformations are applied to day-of-year values
        in this step, but it can be extended for outlier removal or
        other domain-specific logic.

        :return: None
        :rtype: None
        """
        pass  # pragma: no cover

    def scale_second(self) -> None:
        """
        Optionally apply a sinusoidal or cosinusoidal transformation to the day-of-year values.

        If ``"convert"`` is specified as either ``"sine"`` or ``"cosine"`` in :attr:`feature_info`,
        transforms each day-of-year value into a cyclical feature in the range [0, 1].

        :return: None
        :rtype: None
        """

        dispatcher = {"sine": self.convert_sine, "cosine": self.convert_cosine}
        if (self.feature_info is not None) and ("convert" in self.feature_info):
            conversion_type = self.feature_info.get("convert")
            if conversion_type in dispatcher:
                dispatcher[conversion_type]()

    def convert_sine(self) -> None:
        """
        Optionally apply a sinusoidal transformation to the day-of-year values.

        The transformation formula used is:

        .. math::
            day\\_of\\_year_{transformed} = \\frac{{\\sin((day\\_of\\_year - 1) \\cdot 2 \\cdot \\pi / 364) + 1}}{2}

        :return: None
        :rtype: None
        """

        self.features = self.features.with_columns(
            (((pl.col("day_of_year") - 1) * 2 * np.pi / 364).sin() + 1) / 2
        )

    def convert_cosine(self) -> None:
        """
        Optionally apply a cosinusoidal transformation to the day-of-year values.

        The transformation formula used is:

        .. math::
            day\\_of\\_year_{transformed} = \\frac{{\\cos((day\\_of\\_year - 1) \\cdot 2 \\cdot \\pi / 364) + 1}}{2}

        :return: None
        :rtype: None
        """
        self.features = self.features.with_columns(
            (((pl.col("day_of_year") - 1) * 2 * np.pi / 364).cos() + 1) / 2
        )
