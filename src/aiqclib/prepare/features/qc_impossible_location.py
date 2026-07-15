"""
This module defines QCImpossibleLocation, the RTQC3 impossible location test.

The observation latitude and longitude must be sensible: latitude within
[-90, 90] and longitude within [-180, 180]. Null coordinates fail. The test
is profile-level: the flag applies to every observation of the profile
through the shared position columns.
"""

import polars as pl

from aiqclib.prepare.features.qc_item_base import QCItemFeatureBase


class QCImpossibleLocation(QCItemFeatureBase):
    """
    RTQC3 impossible location test (profile-level).

    Fails when latitude or longitude is null or outside the configured
    bounds. Produces the single column ``qc_impossible_location``.
    """

    item_name: str = "impossible_location"
    default_params: dict = {
        "lat_min": -90.0,
        "lat_max": 90.0,
        "lon_min": -180.0,
        "lon_max": 180.0,
    }

    def compute_flags(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Flag observations whose position is not sensible.

        :param df: The observations to check.
        :type df: pl.DataFrame
        :return: Observation keys plus the ``qc_impossible_location`` column.
        :rtype: pl.DataFrame
        """
        lat = pl.col("latitude")
        lon = pl.col("longitude")

        fail = (
            lat.is_null()
            | lon.is_null()
            | ~lat.is_between(self.params["lat_min"], self.params["lat_max"])
            | ~lon.is_between(self.params["lon_min"], self.params["lon_max"])
        )
        return self._select_flags(df, [self._flag_expr(fail, self.flag_column_name())])
