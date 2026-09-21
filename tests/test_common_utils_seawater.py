"""Unit tests for the EOS-80 seawater routines (``common.utils.seawater``).

The reference values are the official UNESCO 1983 (Fofonoff & Millard)
check values plus a handful of oceanographic sanity checks for the three
target regions of the NRT QC module (Arctic, Baltic, Mediterranean).
"""

import warnings

import numpy as np
import polars as pl
import pytest

from aiqclib.common.utils.seawater import (
    adiabatic_lapse_rate,
    brunt_vaisala_squared,
    density_at_surface,
    depth_from_pressure,
    gravity,
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


class TestInputDomain:
    """Values that cannot be measurements are rejected rather than computed."""

    @pytest.mark.parametrize(
        "s, t, p",
        [
            (-9.0, 5.0, 10.0),  # negative salinity
            (-999.0, 5.0, 10.0),  # placeholder salinity
            (35.0, -999.0, 10.0),  # placeholder temperature
            (35.0, 9999.0, 10.0),  # placeholder temperature, positive
            (9.96921e36, 9.96921e36, 10.0),  # netCDF fill value
            (35.0, 5.0, -9999.0),  # placeholder pressure
            (35.0, np.inf, 10.0),  # infinity
        ],
    )
    def test_out_of_domain_is_nan(self, s, t, p):
        """An impossible input yields NaN, not a finite density."""
        assert np.isnan(sigma0(s, t, p))

    @pytest.mark.parametrize(
        "s, t, p",
        [
            (-999.0, 5.0, 10.0),
            (35.0, -999.0, 10.0),
            (9.96921e36, 9.96921e36, 10.0),
        ],
    )
    def test_out_of_domain_is_quiet(self, s, t, p):
        """Rejecting an input raises no numpy overflow or invalid warning."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert np.isnan(sigma0(np.array([s]), np.array([t]), np.array([p]))[0])

    def test_only_the_bad_element_is_lost(self):
        """Neighbouring observations in the same array are unaffected."""
        result = sigma0(
            np.array([35.0, -999.0, 35.0]),
            np.array([5.0, 5.0, 25.0]),
            np.zeros(3),
        )
        assert result[0] == pytest.approx(27.67547, abs=1e-5)
        assert np.isnan(result[1])
        assert result[2] == pytest.approx(23.34306, abs=1e-5)

    @pytest.mark.parametrize(
        "s, t, p",
        [
            (0.0, -2.5, 0.0),  # fresh water below the global range minimum
            (41.0, 40.0, 0.0),  # the global range extremes
            (34.9, -1.9, 11000.0),  # the deepest cold ocean
        ],
    )
    def test_extremes_of_the_qc_ranges_still_compute(self, s, t, p):
        """The bounds are wider than the QC range tests, so nothing real is lost."""
        assert np.isfinite(sigma0(s, t, p))


class TestGravityAndDepth:
    """Gravity and the pressure-to-depth conversion (UNESCO 1983)."""

    def test_gravity_at_the_equator(self):
        assert gravity(0.0) == pytest.approx(9.780318, abs=1e-6)

    def test_gravity_at_the_pole(self):
        assert gravity(90.0) == pytest.approx(9.832177, abs=1e-6)

    def test_gravity_is_symmetric_about_the_equator(self):
        assert gravity(-45.0) == pytest.approx(gravity(45.0))

    def test_depth_check_value(self):
        """DEPTH(10000 dbar, 30 deg) = 9712.653 m."""
        assert depth_from_pressure(10000.0, 30.0) == pytest.approx(9712.653, abs=1e-3)

    def test_surface_pressure_is_the_surface(self):
        assert depth_from_pressure(0.0, 45.0) == pytest.approx(0.0)

    def test_depth_is_close_to_pressure_in_the_upper_ocean(self):
        """A decibar is very nearly a metre, which is why the two get mixed up."""
        assert depth_from_pressure(1000.0, 45.0) == pytest.approx(989.5, abs=1.0)

    @pytest.mark.parametrize(
        "p, lat",
        [(-9999.0, 45.0), (1000.0, -999.0), (1000.0, 91.0), (9.96921e36, 45.0)],
    )
    def test_out_of_domain_is_nan(self, p, lat):
        assert np.isnan(depth_from_pressure(p, lat))

    def test_out_of_domain_is_quiet(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = depth_from_pressure(np.array([9.96921e36]), np.array([45.0]))
            assert np.isnan(result[0])


class TestBruntVaisala:
    """Vertical density stability."""

    def test_stable_column_is_positive(self):
        """Density increasing downward resists being moved, so N^2 > 0."""
        assert brunt_vaisala_squared(27.0, 0.01, 45.0) > 0.0

    def test_inverted_column_is_negative(self):
        assert brunt_vaisala_squared(27.0, -0.01, 45.0) < 0.0

    def test_mixed_layer_is_zero(self):
        assert brunt_vaisala_squared(27.0, 0.0, 45.0) == pytest.approx(0.0)

    def test_magnitude_is_oceanographically_plausible(self):
        """A strong thermocline is of order 1e-4 per second squared."""
        value = brunt_vaisala_squared(27.0, 0.01, 45.0)
        assert 1e-5 < value < 1e-3

    def test_sign_flips_with_the_gradient_only(self):
        up = brunt_vaisala_squared(27.0, 0.01, 45.0)
        down = brunt_vaisala_squared(27.0, -0.01, 45.0)
        assert up == pytest.approx(-down)

    @pytest.mark.parametrize(
        "sigma, gradient, lat",
        [(np.nan, 0.01, 45.0), (27.0, np.nan, 45.0), (27.0, 0.01, 999.0)],
    )
    def test_missing_input_is_nan(self, sigma, gradient, lat):
        assert np.isnan(brunt_vaisala_squared(sigma, gradient, lat))

    def test_impossible_density_is_nan_not_a_division_by_zero(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = brunt_vaisala_squared(
                np.array([-2000.0]), np.array([0.01]), np.array([45.0])
            )
            assert np.isnan(result[0])
