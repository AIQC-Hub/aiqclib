"""
This module defines QCPositionOnLand, the RTQC4 position on land test.

The recommendation checks the profile position against a bathymetry file
(ETOPO2) to confirm it lies in an ocean. This implementation does not read
an external grid: it reads a column already present in the input, holding
the sea floor depth at the position, which is what a bathymetry lookup
would have produced. Computing that column is the caller's job, done
upstream where the input is assembled. Inputs without it cannot run the
item, which is why it is absent from the configuration templates and why a
missing column is an error rather than a silent pass.

The column defaults to ``bathymetry`` rather than ``depth`` on purpose: it
holds the depth of the sea floor at the position, not the depth of the
measurement, and the two are easy to confuse where both exist.

The test is profile-level: the flag applies to every observation of the
profile through the shared position columns.
"""

from typing import Tuple

import polars as pl

from aiqclib.prepare.features.qc_item_base import QCItemFeatureBase


class QCPositionOnLand(QCItemFeatureBase):
    """
    RTQC4 position on land test (profile-level).

    Fails where the depth at the profile position says the position is not
    in the ocean. Produces the single column ``qc_position_on_land``.

    Two parameters describe the column, because both conventions are common
    and reading one as the other inverts the test completely:

    - ``depth_column`` names the column holding the sea floor depth at the
      profile position (default ``"bathymetry"``). This is a separate,
      externally computed column, not an observation depth.
    - ``positive_depth`` says which sign means deeper. With :obj:`True`
      (the default) depths are positive and larger means deeper, so the
      ocean is ``depth > 0``. With :obj:`False` depths are negative below
      sea level, so the ocean is ``depth < 0``.

    Sea level itself (``0``) counts as land under both conventions: it is
    the shoreline, not a position in the ocean.
    """

    item_name: str = "position_on_land"
    default_params: dict = {
        "depth_column": "bathymetry",
        "positive_depth": True,
    }
    scalar_param_names: Tuple[str, ...] = ("depth_column", "positive_depth")

    def compute_flags(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Flag observations whose position is not in the ocean.

        A null depth passes. It means the bathymetry is unknown at that
        position, which is not evidence of land, and matches how every
        other item treats a value it cannot evaluate.

        :param df: The observations to check.
        :type df: pl.DataFrame
        :return: Observation keys plus the ``qc_position_on_land`` column.
        :rtype: pl.DataFrame
        :raises ValueError: If the configured depth column is not in ``df``.
        """
        column = self.params["depth_column"]
        if column not in df.columns:
            raise ValueError(
                f"QC item '{self.item_name}' needs the sea floor depth "
                f"column '{column}', which is not in the input. This column "
                f"is computed externally, before the workflow runs, and is "
                f"not the depth of the measurement. Set 'depth_column' "
                f"under the item's params to the column holding it, or "
                f"remove the item from the QC item set."
            )

        depth = pl.col(column)
        fail = depth <= 0 if self.params["positive_depth"] else depth >= 0
        return self._select_flags(df, [self._flag_expr(fail, self.flag_column_name())])
