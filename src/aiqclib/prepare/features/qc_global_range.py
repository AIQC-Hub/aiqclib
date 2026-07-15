"""
This module defines QCGlobalRange, the RTQC6 global range test.

A gross filter on observed values that must accommodate all expected ocean
extremes: temperature within [-2.5, 40.0] degC and salinity within
[2.0, 41.0] by default. Each configured variable is tested independently
and produces its own flag column. Null values pass (a missing measurement
cannot be range-checked).
"""

from typing import Dict, List

import polars as pl

from aiqclib.prepare.features.qc_item_base import QCItemFeatureBase


class QCGlobalRange(QCItemFeatureBase):
    """
    RTQC6 global range test (observation-level, per variable).

    Fails when a value lies outside the variable's [min, max] range
    (bounds inclusive). Produces one ``{variable}_qc_global_range``
    column per configured variable.
    """

    item_name: str = "global_range"
    default_params: Dict = {
        "temp": {"min": -2.5, "max": 40.0},
        "psal": {"min": 2.0, "max": 41.0},
    }

    def _range_bounds(self, variable: str) -> Dict:
        """
        Return the validated {min, max} bounds for a variable.

        :param variable: The variable to look up.
        :type variable: str
        :return: The bounds dictionary with ``min`` and ``max`` keys.
        :rtype: Dict
        :raises ValueError: If the variable has no complete min/max entry.
        """
        bounds = self.params.get(variable)
        if not isinstance(bounds, dict) or not {"min", "max"} <= set(bounds):
            raise ValueError(
                f"QC item '{self.item_name}' requires 'min' and 'max' "
                f"params for variable '{variable}'."
            )
        return bounds

    def compute_flags(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Flag values outside their variable's configured range.

        :param df: The observations to check.
        :type df: pl.DataFrame
        :return: Observation keys plus one flag column per variable.
        :rtype: pl.DataFrame
        :raises ValueError: If no variables are configured for the item.
        """
        exprs: List[pl.Expr] = []
        for variable in self.get_variables():
            bounds = self._range_bounds(variable)
            fail = ~pl.col(variable).is_between(bounds["min"], bounds["max"])
            exprs.append(self._flag_expr(fail, self.flag_column_name(variable)))

        if not exprs:
            raise ValueError(
                f"QC item '{self.item_name}' has no variables to check; "
                "supply per-variable ranges in params."
            )
        return self._select_flags(df, exprs)
