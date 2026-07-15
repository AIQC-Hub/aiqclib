"""Unit tests for the EOS-80 seawater routines (``common.utils.seawater``).

The reference values are the official UNESCO 1983 (Fofonoff & Millard)
check values plus a handful of oceanographic sanity checks for the three
target regions of the NRT QC module (Arctic, Baltic, Mediterranean).
"""

import numpy as np
import polars as pl
import pytest

from aiqclib.common.utils.seawater import (
    adiabatic_lapse_rate,
    density_at_surface,
    potential_temperature,
    sigma0,
)


class TestUnescoCheckValues:
    """Official UNESCO 1983 check values."""

    def test_adiabatic_lapse_rate(self):
        """ATG(40, 40, 10000) = 3.255976e-4 °C/dbar."""
        assert adiabatic_lapse_rate(40.0, 40.0, 10000.0) == pytest.approx(
            3.255976e-4, abs=1e-10
        )

    def test_potential_temperature(self):
        """theta(40, 40, 10000, 0) = 36.89073 °C."""
        assert potential_temperature(40.0, 40.0, 10000.0, 0.0) == pytest.approx(
            36.89073, abs=1e-5
        )

    @pytest.mark.parametrize(
        "s, t, expected",
        [
            (0.0, 5.0, 999.96675),
            (35.0, 5.0, 1027.67547),
            (35.0, 25.0, 1023.34306),
        ],
    )
    def test_density_at_surface(self, s, t, expected):
        """rho(S, T, 0) matches the published check values (kg/m³)."""
        assert density_at_surface(s, t) == pytest.approx(expected, abs=1e-5)

    def test_sigma0_at_surface(self):
        """At p=0, sigma0 is exactly the surface density anomaly."""
        assert sigma0(35.0, 5.0, 0.0) == pytest.approx(27.67547, abs=1e-5)


class TestPhysicalBehaviour:
    """Oceanographic sanity checks."""

    def test_potential_temperature_below_insitu(self):
        """At depth, theta is slightly below the in-situ temperature."""
        theta = potential_temperature(35.0, 5.0, 5000.0)
        assert 4.0 < float(theta) < 5.0

    def test_potential_temperature_identity_at_surface(self):
        """At the reference pressure, theta equals the temperature."""
        assert potential_temperature(35.0, 10.0, 0.0) == pytest.approx(10.0, abs=1e-12)

    def test_sigma0_increases_with_salinity(self):
        """Saltier water is denser at equal temperature."""
        assert float(sigma0(36.0, 10.0, 0.0)) > float(sigma0(34.0, 10.0, 0.0))

    def test_sigma0_decreases_with_temperature(self):
        """Warmer water is lighter at equal salinity (above ~4 °C)."""
        assert float(sigma0(35.0, 20.0, 0.0)) < float(sigma0(35.0, 10.0, 0.0))

    @pytest.mark.parametrize(
        "s, t, low, high",
        [
            (7.0, 10.0, 4.0, 7.0),  # Baltic surface water
            (38.5, 14.0, 28.0, 30.0),  # Mediterranean intermediate water
            (34.9, -1.5, 27.0, 29.0),  # Arctic cold halocline water
        ],
    )
    def test_regional_sigma0_ranges(self, s, t, low, high):
        """sigma0 lands in the expected range for regional water masses."""
        value = float(sigma0(s, t, 0.0))
        assert low < value < high


class TestVectorisation:
    """Array and polars Series handling."""

    def test_numpy_arrays(self):
        """Arrays in, arrays of the same shape out."""
        s = np.array([35.0, 35.0, 0.0])
        t = np.array([5.0, 25.0, 5.0])
        p = np.zeros(3)
        result = sigma0(s, t, p)
        assert result.shape == (3,)
        assert result[0] == pytest.approx(27.67547, abs=1e-5)

    def test_polars_series_with_null(self):
        """polars Series are accepted; nulls propagate as NaN."""
        s = pl.Series([35.0, None])
        t = pl.Series([5.0, 5.0])
        p = pl.Series([0.0, 0.0])
        result = sigma0(s, t, p)
        assert result[0] == pytest.approx(27.67547, abs=1e-5)
        assert np.isnan(result[1])

    def test_inputs_not_mutated(self):
        """potential_temperature must not modify its input arrays."""
        t = np.array([5.0, 10.0])
        p = np.array([1000.0, 2000.0])
        potential_temperature(np.array([35.0, 35.0]), t, p)
        assert t.tolist() == [5.0, 10.0]
        assert p.tolist() == [1000.0, 2000.0]
