"""
This module defines QCStuckValue, the RTQC13 stuck value test.

The test looks for all measurements of a variable in a profile being
identical, which indicates a stuck sensor. When it occurs, every value of
the affected variable in the profile is flagged. Profiles with fewer than
``min_observations`` non-null measurements are exempt (a single observation
cannot be "stuck"), as are profiles where the variable was not measured.
"""

from typing import Dict, List, Tuple

import polars as pl

from aiqclib.prepare.features.qc_item_base import PROFILE_KEYS, QCItemFeatureBase


class QCStuckValue(QCItemFeatureBase):
    """
    RTQC13 stuck value test (profile-level, per variable).

    Fails a whole profile's variable when all its non-null measurements
    are identical and there are at least ``min_observations`` of them.
    Produces one ``{variable}_qc_stuck_value`` column per configured
    variable. The variables default to temp and psal; the empty per-variable
    dictionaries in the defaults act as enable markers.
    """

    item_name: str = "stuck_value"
    default_params: Dict = {
        "temp": {},
        "psal": {},
        "min_observations": 2,
    }
    scalar_param_names: Tuple[str, ...] = ("min_observations",)

    def compute_flags(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Flag profiles whose variable is stuck at a single value.

        :param df: The observations to check.
        :type df: pl.DataFrame
        :return: Observation keys plus one flag column per variable.
        :rtype: pl.DataFrame
        """
        min_observations = self.params["min_observations"]

        exprs: List[pl.Expr] = []
        for variable in self.get_variables():
            non_null_count = pl.col(variable).count().over(PROFILE_KEYS)
            unique_count = pl.col(variable).drop_nulls().n_unique().over(PROFILE_KEYS)
            fail = (non_null_count >= min_observations) & (unique_count == 1)
            exprs.append(self._flag_expr(fail, self.flag_column_name(variable)))

        return self._select_flags(df, exprs)
