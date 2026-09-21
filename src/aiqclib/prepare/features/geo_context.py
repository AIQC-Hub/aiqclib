"""
This module defines GeoContext, the geographic half of the spatiotemporal
feature group.

``location`` and ``day_of_year`` already say where and when a profile was
taken. What they cannot say is what kind of place that is. The same
temperature at 50 metres means one thing over a 60 metre shelf, where the
measurement is nearly on the bottom and the water column is mixed by tides,
and another over a 4000 metre basin, where it is in the surface layer. The
columns here carry that distinction.

External grids are not read. ``bathymetry`` and ``coast_distance`` are
expected as columns of the input, exactly as
:class:`~aiqclib.prepare.features.qc_position_on_land.QCPositionOnLand`
expects the sea floor depth: sampling GEBCO and GSHHG is an upstream job,
done where the input is assembled, and keeping it there keeps netCDF and
shapefile libraries, their data files, and their coordinate conventions out
of this library and its tests.
"""

import warnings
from typing import Dict, List, Tuple

import polars as pl

from aiqclib.common.constants import OBSERVATION_KEYS
from aiqclib.common.utils.profile_signal import central_difference_expr
from aiqclib.common.utils.seawater import depth_from_pressure, sigma0
from aiqclib.prepare.features.profile_feature_base import ProfileFeatureBase

#: Internal columns used while computing, never emitted.
_DEPTH_COLUMN = "_geo_depth"
_SEAFLOOR_COLUMN = "_geo_seafloor"
_SIGMA_COLUMN = "_geo_sigma0"
_GRADIENT_COLUMN = "_geo_gradient"

#: Outputs that cannot be computed without the bathymetry column.
BATHYMETRY_OUTPUTS: Tuple[str, ...] = (
    "bathymetry",
    "normalized_depth",
    "distance_to_bottom",
    "deep_stable_layer",
)


class GeoContext(ProfileFeatureBase):
    """
    Where in the ocean, and where in the water column, a level sits.

    Outputs, none of them per variable:

    - ``bathymetry``: the sea floor depth at the profile position, in
      metres and positive downward whatever convention the input column
      uses.
    - ``coast_distance``: the distance to the nearest coast, passed
      through from its input column.
    - ``normalized_depth``: the level's depth divided by the sea floor
      depth, so 0 is the surface and 1 the bottom regardless of how deep
      the water is. This is what makes a shelf cast and a deep cast
      comparable.
    - ``distance_to_bottom``: the metres of water beneath the level, which
      ``normalized_depth`` cannot express: 10 metres off the bottom is 10
      metres off the bottom in either place.
    - ``deep_stable_layer``: 1 where the level is deeper than
      ``deep_threshold`` and the water column there is stable but only
      weakly stratified, that is, the quiet deep water where a measurement
      has little excuse to move. Sharp changes are ordinary in the surface
      layer and suspicious here, which is the distinction the flag carries.

    Parameters: ``bathymetry_column`` (``bathymetry``),
    ``coast_distance_column`` (``coast_distance``), ``positive_depth``
    (:obj:`True`, meaning larger values in the bathymetry column are
    deeper), ``deep_threshold`` (1000.0 metres),
    ``stability_threshold`` (0.005 kg/m⁴, the density gradient below which
    a stable column counts as weakly stratified), the input column names
    shared with the other physics features, and ``required``.

    A configured column that is missing from the input is an error, as it
    is for the position-on-land QC item. Setting ``required`` to
    :obj:`False` emits the columns that depend on it as null instead, with
    a warning; that keeps the training frame's schema the same whether or
    not the dataset carries bathymetry, which matters when one
    configuration is run over several regions.
    """

    feature_name: str = "geo_context"
    supported_outputs: Tuple[str, ...] = (
        "bathymetry",
        "coast_distance",
        "normalized_depth",
        "distance_to_bottom",
        "deep_stable_layer",
    )
    default_params: Dict = {
        "bathymetry_column": "bathymetry",
        "coast_distance_column": "coast_distance",
        "positive_depth": True,
        "deep_threshold": 1000.0,
        "stability_threshold": 0.005,
        "salinity_column": "psal",
        "temperature_column": "temp",
        "pressure_column": "pres",
        "latitude_column": "latitude",
        "sigma0_column": None,
        "min_depth_separation": 0.01,
        "required": True,
    }

    def _check_optional_column(self, df: pl.DataFrame, column: str) -> bool:
        """
        Whether an optional input column is usable, warning when it is not.

        :param df: The input observations.
        :type df: pl.DataFrame
        :param column: The configured column name.
        :type column: str
        :return: ``True`` when the column is present.
        :rtype: bool
        :raises ValueError: If the column is absent and ``required`` is set.
        """
        if column in df.columns:
            return True
        if self.params["required"]:
            self.require_columns(df, [column])
        warnings.warn(
            f"Feature '{self.feature_name}' has no column '{column}' in the "
            f"input; the outputs that need it are emitted as null because "
            f"'required' is set to false.",
            stacklevel=2,
        )
        return False

    def _seafloor_expr(self) -> pl.Expr:
        """
        The sea floor depth, as metres positive downward.

        :return: The aliased expression.
        :rtype: pl.Expr
        """
        column = pl.col(self.params["bathymetry_column"])
        depth = column if self.params["positive_depth"] else -column
        # A sea floor at or above sea level is not a sea floor; the
        # position-on-land QC item calls that land under both conventions.
        return pl.when(depth > 0).then(depth).otherwise(None).alias(_SEAFLOOR_COLUMN)

    def _with_water_column(self, df: pl.DataFrame, needs_gradient: bool):
        """
        Add the level depth, and the density gradient when it is needed.

        :param df: The input observations.
        :type df: pl.DataFrame
        :param needs_gradient: Whether ``deep_stable_layer`` was requested.
        :type needs_gradient: pl.DataFrame
        :return: The frame with the internal columns added.
        :rtype: pl.DataFrame
        """
        pressure_column = self.params["pressure_column"]
        latitude_column = self.params["latitude_column"]
        work = df.with_columns(
            pl.Series(depth_from_pressure(df[pressure_column], df[latitude_column]))
            .alias(_DEPTH_COLUMN)
            .fill_nan(None)
        )
        if not needs_gradient:
            return work

        sigma_column = self.params["sigma0_column"]
        if sigma_column is not None:
            density = work[sigma_column].cast(pl.Float64)
        else:
            density = pl.Series(
                sigma0(
                    work[self.params["salinity_column"]],
                    work[self.params["temperature_column"]],
                    work[pressure_column],
                )
            )
        work = work.with_columns(density.alias(_SIGMA_COLUMN).fill_nan(None))
        return work.with_columns(
            central_difference_expr(
                _SIGMA_COLUMN,
                _DEPTH_COLUMN,
                min_separation=self.params["min_depth_separation"],
            ).alias(_GRADIENT_COLUMN)
        )

    def _output_exprs(self, have_bathymetry: bool, have_coast: bool) -> List[pl.Expr]:
        """
        Build the requested output expressions.

        :param have_bathymetry: Whether the bathymetry column is present.
        :type have_bathymetry: bool
        :param have_coast: Whether the coast distance column is present.
        :type have_coast: bool
        :return: One aliased expression per requested output.
        :rtype: List[pl.Expr]
        """
        seafloor = pl.col(_SEAFLOOR_COLUMN)
        depth = pl.col(_DEPTH_COLUMN)
        missing = pl.lit(None, dtype=pl.Float64)

        available: Dict[str, pl.Expr] = {
            "bathymetry": seafloor if have_bathymetry else missing,
            "coast_distance": (
                pl.col(self.params["coast_distance_column"]) if have_coast else missing
            ),
            "normalized_depth": (depth / seafloor) if have_bathymetry else missing,
            "distance_to_bottom": (seafloor - depth) if have_bathymetry else missing,
            "deep_stable_layer": (
                self._deep_stable_expr() if have_bathymetry else missing
            ),
        }
        return [available[output].alias(output) for output in self.outputs]

    def _deep_stable_expr(self) -> pl.Expr:
        """
        The deep, weakly stratified layer flag.

        Stable and weakly stratified means the density gradient is not
        negative (the column is not inverted) and not large (the level is
        not in a pycnocline), which together describe the quiet deep water.

        :return: The flag expression.
        :rtype: pl.Expr
        """
        gradient = pl.col(_GRADIENT_COLUMN)
        deep = pl.col(_DEPTH_COLUMN) >= self.params["deep_threshold"]
        quiet = (gradient >= 0) & (gradient <= self.params["stability_threshold"])
        return (
            pl.when(gradient.is_null() | pl.col(_DEPTH_COLUMN).is_null())
            .then(None)
            .otherwise(deep & quiet)
            .cast(pl.Int8)
        )

    def compute_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Build the requested geographic context columns.

        :param df: The observations of the selected profiles, sorted.
        :type df: pl.DataFrame
        :return: Observation keys plus one column per requested output.
        :rtype: pl.DataFrame
        :raises ValueError: If a required input column is missing.
        """
        needs_bathymetry = any(output in self.outputs for output in BATHYMETRY_OUTPUTS)
        needs_gradient = "deep_stable_layer" in self.outputs
        needed = [self.params["pressure_column"], self.params["latitude_column"]]
        if needs_gradient and self.params["sigma0_column"] is None:
            needed += [
                self.params["salinity_column"],
                self.params["temperature_column"],
            ]
        elif needs_gradient:
            needed.append(self.params["sigma0_column"])
        self.require_columns(df, sorted(set(needed)))

        have_bathymetry = needs_bathymetry and self._check_optional_column(
            df, self.params["bathymetry_column"]
        )
        have_coast = "coast_distance" in self.outputs and self._check_optional_column(
            df, self.params["coast_distance_column"]
        )

        work = self._with_water_column(df, needs_gradient and have_bathymetry)
        if have_bathymetry:
            work = work.with_columns(self._seafloor_expr())

        return work.select(
            [*OBSERVATION_KEYS, *self._output_exprs(have_bathymetry, have_coast)]
        )
