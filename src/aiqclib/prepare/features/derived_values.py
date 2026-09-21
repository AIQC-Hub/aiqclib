"""
This module defines DerivedValues, the computed half of the measured
variables feature group.

Temperature and salinity are measured and reach the models through
``basic_values``. Density is not measured: it is computed from the other
three through the equation of state. The same is true of depth, which a CTD
reports as pressure, and of potential temperature, which is what makes two
parcels at different depths comparable at all.

Everything here comes from :mod:`aiqclib.common.utils.seawater`, so a value
that cannot be a measurement produces null rather than a number (see that
module's input domain section). A dataset that already carries one of these
columns keeps its own: recomputing a quantity someone else derived, and
silently disagreeing with them, helps nobody.
"""

from typing import Dict, List, Tuple

import polars as pl

from aiqclib.common.constants import OBSERVATION_KEYS
from aiqclib.common.utils.seawater import (
    depth_from_pressure,
    potential_temperature,
    sigma0,
)
from aiqclib.prepare.features.profile_feature_base import ProfileFeatureBase


class DerivedValues(ProfileFeatureBase):
    """
    Quantities derived from the measured variables (observation-level).

    Outputs, all in the units of the equation of state:

    - ``sigma0``: the potential density anomaly in kg/m³, the density the
      parcel would have at the surface, less 1000. This is what the feature
      proposal calls density, and what makes two levels comparable without
      the pressure effect swamping the difference between them.
    - ``depth``: metres below the surface, from pressure and latitude.
    - ``potential_temperature``: the temperature the parcel would have at
      the surface, in °C.

    Parameters name the input columns, since the equation of state needs
    specific quantities and only the dataset knows what they are called:
    ``salinity_column`` (``psal``), ``temperature_column`` (``temp``),
    ``pressure_column`` (``pres``) and ``latitude_column`` (``latitude``).

    ``prefer_input_columns`` (default :obj:`True`) makes an output that is
    already a column of the input pass through untouched.
    """

    feature_name: str = "derived_values"
    supported_outputs: Tuple[str, ...] = ("sigma0", "depth", "potential_temperature")
    default_params: Dict = {
        "salinity_column": "psal",
        "temperature_column": "temp",
        "pressure_column": "pres",
        "latitude_column": "latitude",
        "prefer_input_columns": True,
    }

    #: Which input columns each output is computed from.
    _required_params: Dict[str, Tuple[str, ...]] = {
        "sigma0": ("salinity_column", "temperature_column", "pressure_column"),
        "depth": ("pressure_column", "latitude_column"),
        "potential_temperature": (
            "salinity_column",
            "temperature_column",
            "pressure_column",
        ),
    }

    def _compute(self, df: pl.DataFrame, output: str) -> pl.Series:
        """
        Compute one output from the input columns.

        :param df: The observations of the selected profiles.
        :type df: pl.DataFrame
        :param output: The output to compute.
        :type output: str
        :return: The computed column, NaN where it is undefined.
        :rtype: pl.Series
        """
        salinity = df[self.params["salinity_column"]]
        temperature = df[self.params["temperature_column"]]
        pressure = df[self.params["pressure_column"]]

        if output == "sigma0":
            values = sigma0(salinity, temperature, pressure)
        elif output == "depth":
            values = depth_from_pressure(pressure, df[self.params["latitude_column"]])
        else:
            values = potential_temperature(salinity, temperature, pressure, p_ref=0.0)
        return pl.Series(output, values)

    def compute_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Compute the requested derived columns for every row of ``df``.

        :param df: The observations of the selected profiles, sorted.
        :type df: pl.DataFrame
        :return: Observation keys plus one column per requested output.
        :rtype: pl.DataFrame
        :raises ValueError: If an input column a requested output needs is
                            not in the data.
        """
        passthrough: List[str] = []
        computed: List[pl.Series] = []
        needed: List[str] = []

        for output in self.outputs:
            if self.params["prefer_input_columns"] and output in df.columns:
                passthrough.append(output)
                continue
            needed += [self.params[name] for name in self._required_params[output]]

        self.require_columns(df, sorted(set(needed)))

        for output in self.outputs:
            if output not in passthrough:
                computed.append(self._compute(df, output))

        return (
            df.select([*OBSERVATION_KEYS, *passthrough])
            .with_columns(computed)
            # The equation of state marks what it cannot evaluate with NaN;
            # the pipeline carries missing values as null. Passed-through
            # columns are left exactly as the input had them.
            .with_columns(pl.col([series.name for series in computed]).fill_nan(None))
        )
