"""
This module defines ProfileSmooth, the smoothing half of the point-level
feature group.

The idea behind every output here is the same: fit a smooth curve through
the levels around each measurement, then describe how the measurement
relates to it. A profile is a physical object, so the smooth curve is a
fair guess at what the water column is doing; what the measurement does
that the curve does not is the part worth flagging.

That gives three families of column. The curve itself and its slope and
curvature say what the water column is doing. The residual, and its robust
z-score, say how far the measurement sits from it. The spike index and the
curvature-to-residual ratio separate the two cases that a large residual
alone cannot tell apart: a measurement that jumps away from a smooth column
and comes straight back, and a measurement sitting on a real, sharp
feature such as a thermocline.

Savitzky-Golay is used for the smoothing because it is a fixed set of
weights (see :mod:`aiqclib.common.utils.profile_signal`), so it needs no
signal processing dependency and it preserves the height of a peak instead
of flattening it the way a moving average does.
"""

from typing import Dict, List, Tuple

import polars as pl

from aiqclib.common.constants import OBSERVATION_KEYS
from aiqclib.common.utils.profile_signal import (
    fraction_expr,
    index_spacing_expr,
    savgol_expr,
    spike_index_column_expr,
    with_profile_robust_z,
)
from aiqclib.prepare.features.profile_feature_base import ProfileFeatureBase

#: Outputs that are produced once per configured window rather than once.
WINDOWED_OUTPUTS: Tuple[str, ...] = ("outlier_frac", "high_curvature_frac")


class ProfileSmooth(ProfileFeatureBase):
    """
    Savitzky-Golay smoothing and the anomaly scores built on it.

    Per-level outputs, named ``{variable}_{output}``:

    - ``smooth``: the fitted value at the level.
    - ``d1``, ``d2``: the first and second derivative of the fit, per unit
      of ``spacing_column`` (per decibar by default) rather than per level,
      so a profile sampled every 2 db and one sampled every 10 db give
      comparable gradients.
    - ``residual``: the measurement less the fit.
    - ``robust_z``: the residual standardised by the profile's own median
      and median absolute deviation, so it is comparable between profiles
      and between variables without any unit knowledge.
    - ``curvature_ratio``: ``|d2| / |residual|``. A sharp but real feature
      bends the fitted curve, giving a high ratio; a spike leaves the curve
      alone and lands in the residual, giving a low one. This is the pair
      of numbers that tells those two apart.
    - ``spike_index``: the Argo RTQC9 stencil value, unthresholded.

    Windowed outputs, named ``{variable}_w{window}_{output}``, one set per
    entry of ``windows``:

    - ``outlier_frac``: the fraction of the window whose ``robust_z``
      exceeds ``outlier_z`` in magnitude, that is, how disturbed this part
      of the profile is rather than this one level.
    - ``high_curvature_frac``: the same for the curvature, which says
      whether the level sits in a structured part of the water column.

    Parameters: ``window`` (11) and ``polyorder`` (2) for the fit,
    ``spacing_column`` (``pres``, or :obj:`None` to keep derivatives per
    level), ``min_spacing`` (1e-6) below which the conversion is null
    rather than enormous, ``windows`` (``[5, 11, 21, 41]``), ``outlier_z``
    (3.0), ``high_curvature_z`` (3.0), and ``ratio_floor`` (1e-9), the
    residual magnitude below which ``curvature_ratio`` is null instead of a
    division by nearly zero.
    """

    feature_name: str = "profile_smooth"
    supported_outputs: Tuple[str, ...] = (
        "smooth",
        "d1",
        "d2",
        "residual",
        "robust_z",
        "curvature_ratio",
        "spike_index",
        "outlier_frac",
        "high_curvature_frac",
    )
    default_params: Dict = {
        "window": 11,
        "polyorder": 2,
        "spacing_column": "pres",
        "min_spacing": 1e-6,
        "windows": [5, 11, 21, 41],
        "outlier_z": 3.0,
        "high_curvature_z": 3.0,
        "ratio_floor": 1e-9,
    }

    def _scale_expr(self) -> pl.Expr:
        """
        The per-level step used to turn index derivatives into gradients.

        :return: The step expression, or a literal 1 when derivatives are
                 to stay per level.
        :rtype: pl.Expr
        """
        column = self.params["spacing_column"]
        if column is None:
            return pl.lit(1.0)
        step = index_spacing_expr(column)
        return (
            pl.when(step.abs() > self.params["min_spacing"]).then(step).otherwise(None)
        )

    def _fit_columns(self, variable: str) -> List[pl.Expr]:
        """
        The fit, its derivatives and the residual for one variable.

        :param variable: The input column being smoothed.
        :type variable: str
        :return: One aliased expression per intermediate column.
        :rtype: List[pl.Expr]
        """
        window = self.params["window"]
        polyorder = self.params["polyorder"]
        scale = self._scale_expr()
        smooth = savgol_expr(variable, window, polyorder, 0)
        return [
            smooth.alias(self.output_column_name("smooth", variable)),
            (savgol_expr(variable, window, polyorder, 1) / scale).alias(
                self.output_column_name("d1", variable)
            ),
            (savgol_expr(variable, window, polyorder, 2) / scale**2).alias(
                self.output_column_name("d2", variable)
            ),
            (pl.col(variable) - smooth).alias(
                self.output_column_name("residual", variable)
            ),
            spike_index_column_expr(variable).alias(
                self.output_column_name("spike_index", variable)
            ),
        ]

    def _ratio_expr(self, variable: str) -> pl.Expr:
        """
        The curvature-to-residual ratio for one variable.

        :param variable: The input column being smoothed.
        :type variable: str
        :return: The aliased ratio expression.
        :rtype: pl.Expr
        """
        residual = pl.col(self.output_column_name("residual", variable))
        curvature = pl.col(self.output_column_name("d2", variable))
        return (
            pl.when(residual.abs() > self.params["ratio_floor"])
            .then(curvature.abs() / residual.abs())
            .otherwise(None)
            .alias(self.output_column_name("curvature_ratio", variable))
        )

    def _windowed_columns(self, variable: str) -> List[pl.Expr]:
        """
        The window fractions for one variable, one set per window size.

        :param variable: The input column being smoothed.
        :type variable: str
        :return: One aliased expression per (window, output) pair.
        :rtype: List[pl.Expr]
        """
        exprs: List[pl.Expr] = []
        residual_z = pl.col(self._residual_z_name(variable))
        curvature_z = pl.col(self._curvature_z_name(variable))
        for window in self.params["windows"]:
            if "outlier_frac" in self.outputs:
                exprs.append(
                    fraction_expr(
                        residual_z.abs() > self.params["outlier_z"], window
                    ).alias(self._windowed_name(variable, window, "outlier_frac"))
                )
            if "high_curvature_frac" in self.outputs:
                exprs.append(
                    fraction_expr(
                        curvature_z.abs() > self.params["high_curvature_z"], window
                    ).alias(
                        self._windowed_name(variable, window, "high_curvature_frac")
                    )
                )
        return exprs

    def _windowed_name(self, variable: str, window: int, output: str) -> str:
        """
        The emitted name of a windowed output.

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

    def _residual_z_name(self, variable: str) -> str:
        """
        The column holding the robust z of the residual.

        It is an emitted column when ``robust_z`` was asked for and an
        intermediate otherwise, since the window fractions are counted from
        it either way.

        :param variable: The input column.
        :type variable: str
        :return: The column name.
        :rtype: str
        """
        if "robust_z" in self.outputs:
            return self.output_column_name("robust_z", variable)
        return f"_{variable}_robust_z"

    def _curvature_z_name(self, variable: str) -> str:
        """
        The intermediate column holding the robust z of the curvature.

        :param variable: The input column.
        :type variable: str
        :return: The column name.
        :rtype: str
        """
        return f"_{variable}_curvature_z"

    def _emitted_names(self, variable: str) -> List[str]:
        """
        Every column this feature emits for one variable, in order.

        :param variable: The input column.
        :type variable: str
        :return: The emitted column names.
        :rtype: List[str]
        """
        names = [
            self.output_column_name(output, variable)
            for output in self.outputs
            if output not in WINDOWED_OUTPUTS
        ]
        for window in self.params["windows"]:
            names += [
                self._windowed_name(variable, window, output)
                for output in WINDOWED_OUTPUTS
                if output in self.outputs
            ]
        return names

    def compute_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Smooth every configured variable and build the requested columns.

        :param df: The observations of the selected profiles, sorted.
        :type df: pl.DataFrame
        :return: Observation keys plus the emitted columns.
        :rtype: pl.DataFrame
        :raises ValueError: If a configured input column is missing or no
                            variables were given.
        """
        variables = self.get_variables()
        if not variables:
            raise ValueError(
                f"Feature '{self.feature_name}' needs a 'col_names' list "
                f"naming the variables to smooth."
            )
        needed = list(variables)
        if self.params["spacing_column"] is not None:
            needed.append(self.params["spacing_column"])
        self.require_columns(df, needed)

        work = df
        emitted: List[str] = []
        for variable in variables:
            work = work.with_columns(self._fit_columns(variable))
            work = work.with_columns(self._ratio_expr(variable))
            work = with_profile_robust_z(
                work,
                self.output_column_name("residual", variable),
                self._residual_z_name(variable),
            )
            work = with_profile_robust_z(
                work,
                self.output_column_name("d2", variable),
                self._curvature_z_name(variable),
            )
            work = work.with_columns(self._windowed_columns(variable))
            emitted += self._emitted_names(variable)

        return work.select([*OBSERVATION_KEYS, *emitted])
