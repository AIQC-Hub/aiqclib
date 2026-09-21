"""
This module defines RollingStats, the window-level feature group.

The point-level features ask whether one measurement disagrees with the
levels beside it. These ask what the neighbourhood itself looks like: how
warm, how variable, how tightly packed the levels around this one are.
That is the context the point-level scores are judged against, and it is
the reason the proposal asks for several window sizes rather than one. A
narrow window describes the measurement's immediate surroundings and a wide
one the layer it sits in, and a measurement that disagrees with the first
but not the second is a different kind of suspect from one that disagrees
with both.

The robust statistics here, the median absolute deviation and the local
robust z-score, are the ones to reach for rather than the mean and the
standard deviation. A single bad level moves a window's mean and inflates
its standard deviation, so the very thing being looked for corrupts the
yardstick; a median and a median absolute deviation barely notice it.
"""

from typing import Dict, List, Tuple

import polars as pl

from aiqclib.common.constants import OBSERVATION_KEYS
from aiqclib.common.utils.profile_signal import (
    rolling_stat_expr,
    with_rolling_mad,
    with_rolling_robust_z,
)
from aiqclib.prepare.features.profile_feature_base import ProfileFeatureBase

#: Outputs that map straight onto a polars rolling statistic.
SIMPLE_STATS: Tuple[str, ...] = ("mean", "median", "min", "max", "std")


class RollingStats(ProfileFeatureBase):
    """
    Statistics of the window around each level (observation-level).

    Every output is produced once per variable and per window, named
    ``{variable}_w{window}_{output}``:

    - ``mean``, ``median``, ``min``, ``max``, ``std``: the window's
      distribution.
    - ``mad``: its median absolute deviation, the robust counterpart of
      ``std``, computed as the median of the deviations from the window's
      own median.
    - ``robust_z``: the level's own value measured against that median and
      deviation, which is the proposal's "local robust z-score". It is null
      where the window has no spread at all, since a window of identical
      values offers no scale to be unusual against.

    A window of ``w`` points needs ``(w - 1) / 2`` levels on each side, so
    the outputs are null near the ends of a profile, and a window wider
    than the profile produces nothing at all. Setting ``min_samples``
    relaxes that, at the cost of statistics computed from a partial window
    near the surface and the sea floor.

    Parameters: ``windows`` (``[5, 11, 21, 41]``) and ``min_samples``
    (:obj:`None`, meaning the full window).
    """

    feature_name: str = "rolling_stats"
    supported_outputs: Tuple[str, ...] = (
        "mean",
        "median",
        "mad",
        "min",
        "max",
        "std",
        "robust_z",
    )
    default_params: Dict = {
        "windows": [5, 11, 21, 41],
        "min_samples": None,
    }

    def stat_column_name(self, variable: str, window: int, output: str) -> str:
        """
        The emitted name of one window statistic.

        :param variable: The input column.
        :type variable: str
        :param window: The window size.
        :type window: int
        :param output: The short output name.
        :type output: str
        :return: ``{variable}_w{window}_{output}``.
        :rtype: str
        """
        return f"{variable}_w{window}_{output}"

    def compute_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Build every requested statistic for every variable and window.

        :param df: The observations of the selected profiles, sorted.
        :type df: pl.DataFrame
        :return: Observation keys plus the emitted columns.
        :rtype: pl.DataFrame
        :raises ValueError: If no variables were given or an input column
                            is missing.
        """
        variables = self.get_variables()
        if not variables:
            raise ValueError(
                f"Feature '{self.feature_name}' needs a 'col_names' list "
                f"naming the variables to summarise."
            )
        self.require_columns(df, variables)

        min_samples = self.params["min_samples"]
        work = df
        emitted: List[str] = []

        for variable in variables:
            for window in self.params["windows"]:
                simple: List[pl.Expr] = []
                for output in self.outputs:
                    name = self.stat_column_name(variable, window, output)
                    if output in SIMPLE_STATS:
                        simple.append(
                            rolling_stat_expr(
                                variable,
                                window,
                                output,
                                min_samples=min_samples,
                            ).alias(name)
                        )
                    elif output == "mad":
                        work = with_rolling_mad(
                            work, variable, window, name, min_samples=min_samples
                        )
                    else:
                        work = with_rolling_robust_z(
                            work, variable, window, name, min_samples=min_samples
                        )
                    emitted.append(name)
                if simple:
                    work = work.with_columns(simple)

        return work.select([*OBSERVATION_KEYS, *emitted])
