"""
This module defines QCImpossibleDate, the RTQC2 impossible date test.

The observation date must be sensible: after the minimum year and not in
the future. Because the input timestamp column is a parsed datetime,
structurally invalid dates cannot be represented and surface as nulls,
which also fail the test. The test is profile-level: the flag applies to
every observation of the profile through the shared timestamp column.
"""

from datetime import datetime, timezone

import polars as pl

from aiqclib.prepare.features.qc_item_base import QCItemFeatureBase


class QCImpossibleDate(QCItemFeatureBase):
    """
    RTQC2 impossible date test (profile-level).

    Fails when the profile timestamp is null, its year is not greater than
    ``min_year`` (default 1950), or it lies in the future at processing
    time. Produces the single column ``qc_impossible_date``.
    """

    item_name: str = "impossible_date"
    default_params: dict = {
        "min_year": 1950,
        "timestamp_column": "profile_timestamp",
    }

    def compute_flags(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Flag observations whose profile timestamp is not sensible.

        :param df: The observations to check.
        :type df: pl.DataFrame
        :return: Observation keys plus the ``qc_impossible_date`` column.
        :rtype: pl.DataFrame
        """
        ts = pl.col(self.params["timestamp_column"])
        now = datetime.now(timezone.utc).replace(tzinfo=None)

        fail = (
            ts.is_null()
            | (ts.dt.year() <= self.params["min_year"])
            | (ts > pl.lit(now))
        )
        return self._select_flags(df, [self._flag_expr(fail, self.flag_column_name())])
