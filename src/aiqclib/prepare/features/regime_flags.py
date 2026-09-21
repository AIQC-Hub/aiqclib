"""
This module defines RegimeFlags, the "where in the water column am I"
half of the point-level feature group.

The same measurement means different things at different places in a
profile. A half degree step is ordinary inside a thermocline and alarming
inside a mixed layer, where by definition the water is uniform. Anomaly
scores computed from neighbouring levels cannot know which of those they
are looking at, so this feature says it explicitly.

Two kinds of answer are produced. A flag says the level sits in a mixed
layer, or in the steepest part of a variable's gradient. A position says
how far the level is from the peak gradient, normalised by the length of
the profile, so a model can learn "just below the thermocline" rather than
only "in or out".

The definitions are choices, not standards. The mixed layer criterion and
the percentile that counts as the steep part are both parameters; the
defaults are the common threshold criterion (0.03 kg/m3 in density, or 0.2
degrees in temperature, from a near-surface reference level).
"""

from typing import Dict, List, Tuple

import polars as pl

from aiqclib.common.constants import OBSERVATION_KEYS, PROFILE_KEYS
from aiqclib.common.utils.profile_signal import central_difference_expr
from aiqclib.common.utils.seawater import sigma0
from aiqclib.prepare.features.profile_feature_base import ProfileFeatureBase

#: Internal column holding the density used by the mixed layer criterion.
_SIGMA_COLUMN = "_regime_sigma0"

#: The mixed layer criteria understood by ``mixed_layer_criterion``.
MIXED_LAYER_CRITERIA: Tuple[str, ...] = ("density", "temperature")


class RegimeFlags(ProfileFeatureBase):
    """
    Which part of the water column a level sits in (observation-level).

    Outputs:

    - ``in_mixed_layer``: a single column (not per variable), 1 above the
      mixed layer depth and 0 below it. A profile that never crosses the
      criterion is mixed throughout, which is a real answer and not a
      missing one, so every level gets 1.
    - ``in_gradient_layer``: per variable, named
      ``{variable}_in_gradient_layer``, 1 where the magnitude of the
      variable's vertical gradient is at or above the
      ``gradient_percentile`` of that profile's gradients. For temperature
      this is the thermocline and for salinity the halocline; the name is
      general because the test is.
    - ``normalized_depth_to_peak_gradient``: per variable, the level's
      pressure less the pressure of the profile's steepest gradient,
      divided by the profile's pressure range. It is 0 at the peak,
      negative above it and positive below, and comparable between a
      50 dbar cast and a 2000 dbar one.

    Parameters: ``mixed_layer_criterion`` (``density`` or
    ``temperature``), ``density_threshold`` (0.03 kg/m3),
    ``temperature_threshold`` (0.2 degC), ``reference_pressure`` (10 dbar,
    the level the criterion measures from, chosen to sit below the diurnal
    surface layer), ``gradient_percentile`` (0.9), the input column names,
    and ``min_pressure_separation`` (0.01) guarding the gradient division.
    """

    feature_name: str = "regime_flags"
    supported_outputs: Tuple[str, ...] = (
        "in_mixed_layer",
        "in_gradient_layer",
        "normalized_depth_to_peak_gradient",
    )
    default_params: Dict = {
        "mixed_layer_criterion": "density",
        "density_threshold": 0.03,
        "temperature_threshold": 0.2,
        "reference_pressure": 10.0,
        "gradient_percentile": 0.9,
        "salinity_column": "psal",
        "temperature_column": "temp",
        "pressure_column": "pres",
        "min_pressure_separation": 0.01,
    }

    def _criterion_column(self, work: pl.DataFrame) -> Tuple[pl.DataFrame, str, float]:
        """
        Resolve the column and threshold the mixed layer is measured with.

        :param work: The observations of the selected profiles.
        :type work: pl.DataFrame
        :return: The frame (with density added when needed), the column
                 name, and the threshold.
        :rtype: Tuple[pl.DataFrame, str, float]
        :raises ValueError: If the criterion is unknown.
        """
        criterion = self.params["mixed_layer_criterion"]
        if criterion == "temperature":
            return (
                work,
                self.params["temperature_column"],
                self.params["temperature_threshold"],
            )
        if criterion != "density":
            raise ValueError(
                f"Unknown mixed layer criterion '{criterion}' for feature "
                f"'{self.feature_name}'. Valid criteria: "
                f"{list(MIXED_LAYER_CRITERIA)}."
            )

        density = pl.Series(
            sigma0(
                work[self.params["salinity_column"]],
                work[self.params["temperature_column"]],
                work[self.params["pressure_column"]],
            )
        )
        work = work.with_columns(density.alias(_SIGMA_COLUMN).fill_nan(None))
        return work, _SIGMA_COLUMN, self.params["density_threshold"]

    def _mixed_layer_expr(self, column: str, threshold: float) -> pl.Expr:
        """
        The in-mixed-layer flag.

        The reference value is taken at the level nearest
        ``reference_pressure``; the mixed layer depth is the shallowest
        pressure at which the criterion column has moved away from it by
        more than ``threshold``.

        :param column: The criterion column.
        :type column: str
        :param threshold: The departure that ends the mixed layer.
        :type threshold: float
        :return: The aliased flag expression.
        :rtype: pl.Expr
        """
        pressure = pl.col(self.params["pressure_column"])
        distance = (pressure - self.params["reference_pressure"]).abs()
        reference = pl.col(column).sort_by(distance).first().over(PROFILE_KEYS)

        departed = (pl.col(column) - reference).abs() > threshold
        depth = (
            pl.when(departed & (pressure > self.params["reference_pressure"]))
            .then(pressure)
            .otherwise(None)
            .min()
            .over(PROFILE_KEYS)
        )
        # A profile that never departs is mixed all the way down, which is
        # an answer rather than a missing value.
        return (
            pl.when(depth.is_null())
            .then(1)
            .otherwise((pressure < depth).cast(pl.Int8))
            .cast(pl.Int8)
            .alias("in_mixed_layer")
        )

    def _gradient_name(self, variable: str) -> str:
        """
        The intermediate column holding a variable's vertical gradient.

        :param variable: The input column.
        :type variable: str
        :return: The column name.
        :rtype: str
        """
        return f"_{variable}_gradient"

    def _variable_columns(self, variable: str) -> List[pl.Expr]:
        """
        The per-variable regime columns.

        :param variable: The input column.
        :type variable: str
        :return: One aliased expression per requested per-variable output.
        :rtype: List[pl.Expr]
        """
        pressure = pl.col(self.params["pressure_column"])
        magnitude = pl.col(self._gradient_name(variable)).abs()
        exprs: List[pl.Expr] = []

        if "in_gradient_layer" in self.outputs:
            cutoff = magnitude.quantile(self.params["gradient_percentile"]).over(
                PROFILE_KEYS
            )
            # The magnitude must also be non-zero. In a profile that is
            # uniform over most of its length the percentile itself is
            # zero, and "at or above zero" would flag the flat part as the
            # steepest part of the water column.
            exprs.append(
                pl.when(magnitude.is_null())
                .then(None)
                .otherwise((magnitude >= cutoff) & (magnitude > 0))
                .cast(pl.Int8)
                .alias(self.output_column_name("in_gradient_layer", variable))
            )

        if "normalized_depth_to_peak_gradient" in self.outputs:
            # Ascending sort with nulls first, so the last entry is the
            # largest magnitude rather than a level whose gradient is
            # unknown.
            peak = (
                pressure.sort_by(magnitude, nulls_last=False).last().over(PROFILE_KEYS)
            )
            span = (pressure.max() - pressure.min()).over(PROFILE_KEYS)
            exprs.append(
                pl.when(span > 0)
                .then((pressure - peak) / span)
                .otherwise(None)
                .alias(
                    self.output_column_name(
                        "normalized_depth_to_peak_gradient", variable
                    )
                )
            )
        return exprs

    def _needs_variables(self) -> bool:
        """
        Whether any requested output is computed per variable.

        :return: ``True`` when a per-variable output was requested.
        :rtype: bool
        """
        return any(
            output in self.outputs
            for output in ("in_gradient_layer", "normalized_depth_to_peak_gradient")
        )

    def compute_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Build the requested regime columns.

        :param df: The observations of the selected profiles, sorted.
        :type df: pl.DataFrame
        :return: Observation keys plus the emitted columns.
        :rtype: pl.DataFrame
        :raises ValueError: If a configured input column is missing, the
                            criterion is unknown, or a per-variable output
                            was requested without ``col_names``.
        """
        variables = self.get_variables()
        if self._needs_variables() and not variables:
            raise ValueError(
                f"Feature '{self.feature_name}' needs a 'col_names' list "
                f"naming the variables to find gradient layers in."
            )

        needed = [self.params["pressure_column"], *variables]
        if "in_mixed_layer" in self.outputs:
            needed.append(self.params["temperature_column"])
            if self.params["mixed_layer_criterion"] == "density":
                needed.append(self.params["salinity_column"])
        self.require_columns(df, sorted(set(needed)))

        work = df
        emitted: List[str] = []

        if "in_mixed_layer" in self.outputs:
            work, column, threshold = self._criterion_column(work)
            work = work.with_columns(self._mixed_layer_expr(column, threshold))
            emitted.append("in_mixed_layer")

        for variable in variables:
            work = work.with_columns(
                central_difference_expr(
                    variable,
                    self.params["pressure_column"],
                    min_separation=self.params["min_pressure_separation"],
                ).alias(self._gradient_name(variable))
            )
            columns = self._variable_columns(variable)
            work = work.with_columns(columns)
            emitted += [expr.meta.output_name() for expr in columns]

        return work.select([*OBSERVATION_KEYS, *emitted])
