"""
This module defines the LocationFeat class, a specialized feature extractor
for geographical coordinates (longitude, latitude) within a specified dataset.

It extends the generic FeatureBase to handle the specific requirements of
location data, including extraction from raw profiles and optional scaling.
"""

from typing import Optional, Dict

import polars as pl

from aiqclib.common.base.feature_base import FeatureBase


class LocationFeat(FeatureBase):
    """
    A feature extraction class designed specifically for location-based fields
    (e.g., longitude, latitude) within the Copernicus CTD dataset.

    This class uses the provided data frames to gather location-related fields
    and optionally apply scaling methods. It inherits from
    :class:`~aiqclib.common.base.feature_base.FeatureBase` which defines a generic
    feature extraction workflow.
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
        Initialize the location feature extractor with relevant data frames.

        :param target_name: The key for the target variable in :attr:`selected_rows`.
        :type target_name: Optional[str]
        :param feature_info: A dictionary describing feature parameters, typically including scaling statistics.
        :type feature_info: Optional[Dict]
        :param selected_profiles: A Polars DataFrame containing a subset of profiles relevant to feature extraction.
        :type selected_profiles: Optional[pl.DataFrame]
        :param filtered_input: A filtered Polars DataFrame of input data.
        :type filtered_input: Optional[pl.DataFrame]
        :param selected_rows: A dictionary mapping target names to Polars DataFrames of relevant rows.
        :type selected_rows: Optional[Dict[str, pl.DataFrame]]
        :param summary_stats: A Polars DataFrame containing statistical information for scaling.
        :type summary_stats: Optional[pl.DataFrame]
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
        Gather and merge location columns (e.g., longitude and latitude) from
        :attr:`selected_profiles` into :attr:`selected_rows` to form the final
        feature set in :attr:`features`.

        :returns: None. The result is stored in the :attr:`features` attribute.
        :rtype: None
        """
        self.features = (
            self.selected_rows[self.target_name]
            .select(["row_id", "platform_code", "profile_no"])
            .join(
                self.selected_profiles.select(
                    ["platform_code", "profile_no", "longitude", "latitude"]
                ).unique(),
                on=["platform_code", "profile_no"],
                how="left",
            )
            .drop(["platform_code", "profile_no"])
        )

    def scale_first(self) -> None:
        """
        Initial scaling or normalization procedure (currently unimplemented).

        :returns: None.
        :rtype: None
        """
        pass  # pragma: no cover

    def scale_second(self) -> None:
        """
        Apply a min-max scaling pass to each feature (column) specified in
        :attr:`feature_info["stats"]`.

        :returns: None. Scaling is applied in-place to the :attr:`features` DataFrame.
        :rtype: None
        """
        if self.feature_info["stats_set"]["type"] == "min_max":
            for k, v in self.feature_info["stats"].items():
                self.features = self.features.with_columns(
                    ((pl.col(k) - v["min"]) / (v["max"] - v["min"])).alias(k)
                )
