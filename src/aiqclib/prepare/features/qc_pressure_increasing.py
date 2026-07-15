"""
This module defines QCPressureIncreasing, the RTQC8 pressure increasing test.

The profile must have monotonically increasing pressures in observation
order. In a region of constant pressure, all but the first of the
consecutive constant pressures are flagged. In a region where pressure
reverses, all pressures in the reversed part (below the running maximum of
the preceding pressures) are flagged. Pressure is shared by all variables,
so the test produces a single profile-wide column.
"""

import polars as pl

from aiqclib.prepare.features.qc_item_base import (
    PROFILE_KEYS,
    QCItemFeatureBase,
)


class QCPressureIncreasing(QCItemFeatureBase):
    """
    RTQC8 pressure increasing test (observation-level, variable-independent).

    Fails an observation when its pressure equals the previous pressure
    (constant run) or lies below the running maximum of the preceding
    pressures (reversed segment). Produces the single column
    ``qc_pressure_increasing``.
    """

    item_name: str = "pressure_increasing"

    #: Column defining the vertical order of observations within a profile.
    order_column: str = "observation_no"

    def compute_flags(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Flag constant and reversed pressure regions within each profile.

        :param df: The observations to check.
        :type df: pl.DataFrame
        :return: Observation keys plus the ``qc_pressure_increasing`` column.
        :rtype: pl.DataFrame
        """
        pres = pl.col("pres")
        prev = pres.shift(1).over(PROFILE_KEYS, order_by=self.order_column)
        prev_max = (
            pres.cum_max().shift(1).over(PROFILE_KEYS, order_by=self.order_column)
        )

        # Constant run: all but the first of consecutive equal pressures.
        # Reversal: below the running maximum of the preceding pressures.
        fail = (pres == prev) | (pres < prev_max)
        return self._select_flags(df, [self._flag_expr(fail, self.flag_column_name())])
