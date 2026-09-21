"""
This module defines NeighborDiff, the neighbour comparison half of the
point-level feature group.

``flank_up`` and ``flank_down`` already hand the model the *values* at the
levels above and below. That works, but it makes the model learn the
subtraction itself, and it ties the feature to the absolute value of the
variable: the same 0.5 degree step is a different pair of numbers in the
Baltic than in the Mediterranean. Differencing first removes the baseline
and leaves the part that carries the anomaly.

Differences are taken at several lags because the two failures they catch
look different at different distances. A single bad level stands out at lag
1 and fades as the lag grows; a shifted or drifting sensor shows up as a
step that every lag sees.
"""

from typing import Dict, List, Tuple

import polars as pl

from aiqclib.common.constants import OBSERVATION_KEYS
from aiqclib.common.utils.profile_signal import (
    DIFF_DIRECTIONS,
    fraction_expr,
    neighbor_diff_expr,
    with_profile_robust_z,
)
from aiqclib.prepare.features.profile_feature_base import ProfileFeatureBase


class NeighborDiff(ProfileFeatureBase):
    """
    Differences against the levels above and below (observation-level).

    Outputs:

    - ``diff``: one column per variable, lag and direction, named
      ``{variable}_diff_{direction}_{lag}``. ``up`` subtracts the shallower
      level, ``down`` the deeper one. Levels without that neighbour are
      null, so lag 5 costs five levels at each end of the profile.
    - ``large_diff_frac``: per variable and window, named
      ``{variable}_w{window}_large_diff_frac``, the fraction of the window
      whose reference difference counts as large. This is the window-level
      "% of large neighbor differences".

    What counts as large is the one number the feature proposal leaves
    open, so there are two ways to say it. By default the reference
    difference is standardised against the profile's own median and median
    absolute deviation and compared with ``large_diff_z``, which needs no
    knowledge of the variable's units and adapts to how noisy the profile
    is. Setting ``large_diff_threshold`` instead compares the magnitude
    directly, either as one number for every variable or as a mapping from
    variable name to number.

    Parameters: ``lags`` (``[1, 2, 3, 4, 5]``), ``directions``
    (``["up", "down"]``), ``windows`` (``[5, 11, 21, 41]``),
    ``reference_lag`` (1) and ``reference_direction`` (``up``) naming the
    difference the fraction counts, ``large_diff_z`` (3.0) and
    ``large_diff_threshold`` (:obj:`None`).
    """

    feature_name: str = "neighbor_diff"
    supported_outputs: Tuple[str, ...] = ("diff", "large_diff_frac")
    default_params: Dict = {
        "lags": [1, 2, 3, 4, 5],
        "directions": list(DIFF_DIRECTIONS),
        "windows": [5, 11, 21, 41],
        "reference_lag": 1,
        "reference_direction": "up",
        "large_diff_z": 3.0,
        "large_diff_threshold": None,
    }

    def diff_column_name(self, variable: str, direction: str, lag: int) -> str:
        """
        The emitted name of one difference column.

        :param variable: The input column.
        :type variable: str
        :param direction: ``"up"`` or ``"down"``.
        :type direction: str
        :param lag: The neighbour distance.
        :type lag: int
        :return: ``{variable}_diff_{direction}_{lag}``.
        :rtype: str
        """
        return f"{variable}_diff_{direction}_{lag}"

    def _reference_name(self, variable: str) -> str:
        """
        The difference column the large-difference fraction counts.

        It is one of the emitted columns when that lag and direction were
        asked for, and an intermediate otherwise.

        :param variable: The input column.
        :type variable: str
        :return: The column name.
        :rtype: str
        """
        return self.diff_column_name(
            variable,
            self.params["reference_direction"],
            self.params["reference_lag"],
        )

    def _large_condition(self, variable: str) -> pl.Expr:
        """
        Whether the reference difference at a level counts as large.

        :param variable: The input column.
        :type variable: str
        :return: A boolean expression.
        :rtype: pl.Expr
        """
        threshold = self.params["large_diff_threshold"]
        if threshold is None:
            return (
                pl.col(f"_{variable}_reference_z").abs() > self.params["large_diff_z"]
            )
        if isinstance(threshold, dict):
            threshold = threshold[variable]
        return pl.col(self._reference_name(variable)).abs() > threshold

    def _diff_columns(self, variable: str) -> List[pl.Expr]:
        """
        Every difference column needed for one variable.

        The reference difference is always computed, even when its lag or
        direction was not asked for, because the fraction is counted from
        it; it is simply not emitted.

        :param variable: The input column.
        :type variable: str
        :return: One aliased expression per difference column.
        :rtype: List[pl.Expr]
        """
        wanted = {
            (direction, lag)
            for direction in self.params["directions"]
            for lag in self.params["lags"]
        }
        if "large_diff_frac" in self.outputs:
            wanted.add(
                (self.params["reference_direction"], self.params["reference_lag"])
            )
        return [
            neighbor_diff_expr(variable, lag, direction).alias(
                self.diff_column_name(variable, direction, lag)
            )
            for direction, lag in sorted(wanted)
        ]

    def _emitted_names(self, variable: str) -> List[str]:
        """
        Every column this feature emits for one variable, in order.

        :param variable: The input column.
        :type variable: str
        :return: The emitted column names.
        :rtype: List[str]
        """
        names: List[str] = []
        if "diff" in self.outputs:
            names += [
                self.diff_column_name(variable, direction, lag)
                for direction in self.params["directions"]
                for lag in self.params["lags"]
            ]
        if "large_diff_frac" in self.outputs:
            names += [
                f"{variable}_w{window}_large_diff_frac"
                for window in self.params["windows"]
            ]
        return names

    def compute_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Build the requested difference columns and window fractions.

        :param df: The observations of the selected profiles, sorted.
        :type df: pl.DataFrame
        :return: Observation keys plus the emitted columns.
        :rtype: pl.DataFrame
        :raises ValueError: If no variables were given, an input column is
                            missing, or an absolute threshold mapping has
                            no entry for a configured variable.
        """
        variables = self.get_variables()
        if not variables:
            raise ValueError(
                f"Feature '{self.feature_name}' needs a 'col_names' list "
                f"naming the variables to difference."
            )
        self.require_columns(df, variables)
        self._check_thresholds(variables)

        work = df
        emitted: List[str] = []
        for variable in variables:
            work = work.with_columns(self._diff_columns(variable))
            if "large_diff_frac" in self.outputs:
                if self.params["large_diff_threshold"] is None:
                    work = with_profile_robust_z(
                        work,
                        self._reference_name(variable),
                        f"_{variable}_reference_z",
                    )
                condition = self._large_condition(variable)
                work = work.with_columns(
                    [
                        fraction_expr(condition, window).alias(
                            f"{variable}_w{window}_large_diff_frac"
                        )
                        for window in self.params["windows"]
                    ]
                )
            emitted += self._emitted_names(variable)

        return work.select([*OBSERVATION_KEYS, *emitted])

    def _check_thresholds(self, variables: List[str]) -> None:
        """
        Raise when an absolute threshold mapping is missing a variable.

        :param variables: The configured variables.
        :type variables: List[str]
        :raises ValueError: If a variable has no threshold entry.
        """
        threshold = self.params["large_diff_threshold"]
        if not isinstance(threshold, dict):
            return
        missing = [name for name in variables if name not in threshold]
        if missing:
            raise ValueError(
                f"Feature '{self.feature_name}' has no "
                f"'large_diff_threshold' entry for {missing}. Give one per "
                f"variable, give a single number for all of them, or remove "
                f"the key to use the robust z-score instead."
            )
