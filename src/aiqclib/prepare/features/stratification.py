"""
This module defines Stratification, the vertical density stability feature
group.

A stable water column has denser water underneath lighter water, and the
Brunt-Vaisala frequency squared, N², measures how strongly. It is positive
where the column is stable, near zero in a well mixed layer, and negative
where the column is inverted, which is physically impossible to sustain and
therefore evidence that something in the measurement is wrong.

That makes N² a different kind of feature from a statistical anomaly score:
it carries a physical constraint rather than a comparison with neighbouring
values, which is why the proposal lists it as its own group. The RTQC14
density inversion item tests the same physics as a pass or fail; here the
quantity itself reaches the model, along with its magnitude and a flag.
"""

from typing import Dict, Tuple

import polars as pl

from aiqclib.common.constants import OBSERVATION_KEYS
from aiqclib.common.utils.profile_signal import central_difference_expr
from aiqclib.common.utils.seawater import (
    brunt_vaisala_squared,
    depth_from_pressure,
    sigma0,
)
from aiqclib.prepare.features.profile_feature_base import ProfileFeatureBase

#: Internal column names used while computing, never emitted.
_SIGMA_COLUMN = "_stratification_sigma0"
_DEPTH_COLUMN = "_stratification_depth"
_GRADIENT_COLUMN = "_stratification_gradient"


class Stratification(ProfileFeatureBase):
    """
    Vertical density stability of the water column (observation-level).

    Outputs:

    - ``sigma0_gradient``: the centred vertical gradient of the potential
      density anomaly in kg/m⁴, positive where density increases downward.
    - ``n2``: the Brunt-Vaisala frequency squared in 1/s², positive for a
      stable column and negative for an inverted one.
    - ``n2_abs``: its magnitude, which says how strongly stratified a level
      is without saying in which direction.
    - ``unstable_flag``: 1 where ``n2`` is below ``unstable_n2``, else 0.

    The first and last level of every profile have no pair of neighbours to
    difference across, so all four outputs are null there.

    Parameters name the input columns as in
    :class:`~aiqclib.prepare.features.derived_values.DerivedValues`, and add:

    - ``sigma0_column``: a potential density anomaly column already in the
      input to use instead of recomputing one (default :obj:`None`).
    - ``unstable_n2``: the threshold for ``unstable_flag`` (default ``0.0``,
      that is, any inversion at all).
    - ``min_depth_separation``: metres. A gradient across neighbours closer
      together than this is null rather than a division by nearly zero
      (default ``0.01``).
    """

    feature_name: str = "stratification"
    supported_outputs: Tuple[str, ...] = (
        "sigma0_gradient",
        "n2",
        "n2_abs",
        "unstable_flag",
    )
    default_params: Dict = {
        "salinity_column": "psal",
        "temperature_column": "temp",
        "pressure_column": "pres",
        "latitude_column": "latitude",
        "sigma0_column": None,
        "unstable_n2": 0.0,
        "min_depth_separation": 0.01,
    }

    def _with_density_and_depth(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Add the potential density anomaly and depth the gradient needs.

        :param df: The observations of the selected profiles, sorted.
        :type df: pl.DataFrame
        :return: The frame with the two internal columns added.
        :rtype: pl.DataFrame
        :raises ValueError: If a configured input column is missing.
        """
        pressure_column = self.params["pressure_column"]
        latitude_column = self.params["latitude_column"]
        sigma_column = self.params["sigma0_column"]

        if sigma_column is not None:
            self.require_columns(df, [sigma_column, pressure_column, latitude_column])
            density = df[sigma_column].cast(pl.Float64)
        else:
            self.require_columns(
                df,
                [
                    self.params["salinity_column"],
                    self.params["temperature_column"],
                    pressure_column,
                    latitude_column,
                ],
            )
            density = pl.Series(
                sigma0(
                    df[self.params["salinity_column"]],
                    df[self.params["temperature_column"]],
                    df[pressure_column],
                )
            )

        depth = pl.Series(depth_from_pressure(df[pressure_column], df[latitude_column]))
        return df.with_columns(
            density.alias(_SIGMA_COLUMN).fill_nan(None),
            depth.alias(_DEPTH_COLUMN).fill_nan(None),
        )

    def compute_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Compute the requested stability columns for every row of ``df``.

        :param df: The observations of the selected profiles, sorted.
        :type df: pl.DataFrame
        :return: Observation keys plus one column per requested output.
        :rtype: pl.DataFrame
        :raises ValueError: If a configured input column is missing.
        """
        work = self._with_density_and_depth(df).with_columns(
            central_difference_expr(
                _SIGMA_COLUMN,
                _DEPTH_COLUMN,
                min_separation=self.params["min_depth_separation"],
            ).alias(_GRADIENT_COLUMN)
        )

        n2 = pl.Series(
            brunt_vaisala_squared(
                work[_SIGMA_COLUMN],
                work[_GRADIENT_COLUMN],
                work[self.params["latitude_column"]],
            )
        ).fill_nan(None)
        work = work.with_columns(n2.alias("n2"))

        available = {
            "sigma0_gradient": pl.col(_GRADIENT_COLUMN),
            "n2": pl.col("n2"),
            "n2_abs": pl.col("n2").abs(),
            "unstable_flag": pl.when(pl.col("n2").is_null())
            .then(None)
            .otherwise(pl.col("n2") < self.params["unstable_n2"])
            .cast(pl.Int8),
        }
        return work.select(
            [
                *OBSERVATION_KEYS,
                *[available[output].alias(output) for output in self.outputs],
            ]
        )
