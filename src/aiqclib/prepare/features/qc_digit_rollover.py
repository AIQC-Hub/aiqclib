"""
This module defines QCDigitRollover, the RTQC12 digit rollover test.

Sensors store temperature and salinity in a limited bit range; when the
range is exceeded, stored values roll over to the lower end. The test
detects rollovers that were not compensated for when the profile was
constructed, by flagging differences between vertically adjacent
measurements greater than 10 degC for temperature and 5 for salinity
(by default).
"""

from typing import Dict, List

import polars as pl

from aiqclib.prepare.features.qc_item_base import (
    PROFILE_KEYS,
    QCItemFeatureBase,
)


class QCDigitRollover(QCItemFeatureBase):
    """
    RTQC12 digit rollover test (observation-level, per variable).

    Fails a value when its absolute difference from the previous
    observation exceeds the variable's threshold. Produces one
    ``{variable}_qc_digit_rollover`` column per configured variable;
    the first observation of a profile always passes.
    """

    item_name: str = "digit_rollover"
    default_params: Dict = {
        "temp": 10.0,
        "psal": 5.0,
    }

    #: Column defining the vertical order of observations within a profile.
    order_column: str = "observation_no"

    def compute_flags(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Flag adjacent-difference jumps larger than the variable threshold.

        :param df: The observations to check.
        :type df: pl.DataFrame
        :return: Observation keys plus one flag column per variable.
        :rtype: pl.DataFrame
        :raises ValueError: If a variable has no numeric threshold.
        """
        exprs: List[pl.Expr] = []
        for variable in self.get_variables():
            threshold = self.params.get(variable)
            if not isinstance(threshold, (int, float)):
                raise ValueError(
                    f"QC item '{self.item_name}' requires a numeric "
                    f"threshold for variable '{variable}'."
                )
            value = pl.col(variable)
            prev = value.shift(1).over(PROFILE_KEYS, order_by=self.order_column)
            fail = (value - prev).abs() > threshold
            exprs.append(self._flag_expr(fail, self.flag_column_name(variable)))

        return self._select_flags(df, exprs)
