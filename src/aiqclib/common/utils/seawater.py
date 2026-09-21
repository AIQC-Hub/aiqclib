"""
EOS-80 seawater routines (UNESCO 1983).

Implements the equation of state of seawater from Fofonoff & Millard (1983),
UNESCO Technical Papers in Marine Science #44: the adiabatic lapse rate,
potential temperature, density at atmospheric pressure, and the potential
density anomaly sigma-0 used by the RTQC14 density inversion test, plus the
gravity, depth and stability relations the feature classes need.

All functions follow the UNESCO argument order (salinity, temperature,
pressure) with practical salinity (PSS-78), in-situ temperature in degrees
Celsius (IPTS-68), and pressure in decibars. Inputs may be scalars, numpy
arrays, or polars Series; computation is vectorised with numpy and NaN
values propagate through the results.

Everything here is **elementwise**: a routine sees one water parcel at a
time and never a neighbouring level. Anything that compares levels, such as
the density gradient that :func:`brunt_vaisala_squared` takes as an
argument, belongs in :mod:`aiqclib.common.utils.profile_signal`, where the
profile boundaries are respected.

Inputs outside :data:`SALINITY_LIMITS`, :data:`TEMPERATURE_LIMITS` or
:data:`PRESSURE_LIMITS` are treated as missing and yield NaN. See
`Input domain`_.

.. _Input domain:

Input domain
------------

The routines are polynomial fits to measurements of sea water, so outside
the range of the ocean they produce a number that means nothing. The bounds
below are deliberately far wider than any measurement, so they reject only
what cannot be one: a negative salinity, a placeholder such as -999, or a
netCDF fill value of 9.96921e36.

Rejecting them matters twice over. An unreachable value like the fill value
overflows the fifth-power terms of the density polynomial, which used to
raise a stream of numpy ``RuntimeWarning`` messages during a QC run; and a
merely impossible one like -999 degrees does not overflow, so it used to
yield a finite density hundreds of times denser than sea water, which made
the density inversion test flag the good observation next to it. NaN says
"this cannot be density-checked", which the test already counts as a pass.
"""

from typing import Tuple, Union

import numpy as np
import polars as pl

ArrayLike = Union[float, list, np.ndarray, pl.Series]

#: Practical salinity accepted by the EOS-80 routines (inclusive bounds).
SALINITY_LIMITS: Tuple[float, float] = (0.0, 60.0)

#: Temperature in degrees Celsius accepted by the EOS-80 routines.
TEMPERATURE_LIMITS: Tuple[float, float] = (-10.0, 60.0)

#: Pressure in decibars accepted by the EOS-80 routines (the deepest ocean
#: is near 11,000 dbar).
PRESSURE_LIMITS: Tuple[float, float] = (-10.0, 20000.0)

#: Latitude in degrees accepted by the depth and gravity routines.
LATITUDE_LIMITS: Tuple[float, float] = (-90.0, 90.0)


def _to_array(values: ArrayLike) -> np.ndarray:
    """
    Convert an input to a float64 numpy array (nulls become NaN).

    :param values: A scalar, list, numpy array, or polars Series.
    :type values: ArrayLike
    :return: The values as a float64 numpy array.
    :rtype: numpy.ndarray
    """
    if isinstance(values, pl.Series):
        return values.cast(pl.Float64).to_numpy()
    return np.asarray(values, dtype=np.float64)


def _in_domain(values: ArrayLike, limits: Tuple[float, float]) -> np.ndarray:
    """
    Convert an input to an array, replacing out-of-range values with NaN.

    Infinities and values already NaN fail the comparison too, so the
    result holds only numbers the polynomials can be evaluated for.

    :param values: A scalar, list, numpy array, or polars Series.
    :type values: ArrayLike
    :param limits: The inclusive ``(low, high)`` bounds to keep.
    :type limits: Tuple[float, float]
    :return: The values as a float64 numpy array, out of range as NaN.
    :rtype: numpy.ndarray
    """
    array = _to_array(values)
    low, high = limits
    return np.where((array >= low) & (array <= high), array, np.nan)


def _lapse_rate(s: np.ndarray, t: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
    Evaluate the adiabatic lapse rate polynomial without checking the domain.

    Used by :func:`potential_temperature`, whose Runge-Kutta march feeds
    back intermediate temperatures and pressures: those are results rather
    than inputs and are not range-checked again.

    :param s: Practical salinity (PSS-78).
    :type s: numpy.ndarray
    :param t: In-situ temperature in °C (IPTS-68).
    :type t: numpy.ndarray
    :param p: Pressure in decibars.
    :type p: numpy.ndarray
    :return: Adiabatic lapse rate in °C/dbar.
    :rtype: numpy.ndarray
    """
    ds = s - 35.0
    return (
        (((-2.1687e-16 * t + 1.8676e-14) * t - 4.6206e-13) * p) * p
        + (
            (2.7759e-12 * t - 1.1351e-10) * ds
            + ((-5.4481e-14 * t + 8.733e-12) * t - 6.7795e-10) * t
            + 1.8741e-8
        )
        * p
        + (-4.2393e-8 * t + 1.8932e-6) * ds
        + ((6.6228e-10 * t - 6.836e-8) * t + 8.5258e-6) * t
        + 3.5803e-5
    )


def adiabatic_lapse_rate(s: ArrayLike, t: ArrayLike, p: ArrayLike) -> np.ndarray:
    """
    Adiabatic temperature gradient of seawater (Bryden 1973, UNESCO 1983).

    Check value: ``adiabatic_lapse_rate(40, 40, 10000)`` = 3.255976e-4 °C/dbar.

    :param s: Practical salinity (PSS-78).
    :type s: ArrayLike
    :param t: In-situ temperature in °C (IPTS-68).
    :type t: ArrayLike
    :param p: Pressure in decibars.
    :type p: ArrayLike
    :return: Adiabatic lapse rate in °C/dbar, NaN where an input is missing
             or outside the accepted domain.
    :rtype: numpy.ndarray
    """
    return _lapse_rate(
        _in_domain(s, SALINITY_LIMITS),
        _in_domain(t, TEMPERATURE_LIMITS),
        _in_domain(p, PRESSURE_LIMITS),
    )


def potential_temperature(
    s: ArrayLike, t: ArrayLike, p: ArrayLike, p_ref: float = 0.0
) -> np.ndarray:
    """
    Potential temperature of seawater (Fofonoff & Millard 1983).

    Integrates the adiabatic lapse rate from the in-situ pressure to the
    reference pressure with the standard Runge-Kutta 4 scheme of the UNESCO
    ``PTMP`` routine.

    Check value: ``potential_temperature(40, 40, 10000, 0)`` = 36.89073 °C.

    :param s: Practical salinity (PSS-78).
    :type s: ArrayLike
    :param t: In-situ temperature in °C (IPTS-68).
    :type t: ArrayLike
    :param p: Pressure in decibars.
    :type p: ArrayLike
    :param p_ref: Reference pressure in decibars, defaults to 0 (surface).
    :type p_ref: float
    :return: Potential temperature in °C referenced to ``p_ref``, NaN where
             an input is missing or outside the accepted domain.
    :rtype: numpy.ndarray
    """
    s = _in_domain(s, SALINITY_LIMITS)
    t = _in_domain(t, TEMPERATURE_LIMITS)
    p = _in_domain(p, PRESSURE_LIMITS)

    h = p_ref - p
    xk = h * _lapse_rate(s, t, p)
    t = t + 0.5 * xk
    q = xk
    p = p + 0.5 * h
    xk = h * _lapse_rate(s, t, p)
    t = t + 0.29289322 * (xk - q)
    q = 0.58578644 * xk + 0.121320344 * q
    xk = h * _lapse_rate(s, t, p)
    t = t + 1.707106781 * (xk - q)
    q = 3.414213562 * xk - 4.121320344 * q
    p = p + 0.5 * h
    xk = h * _lapse_rate(s, t, p)
    return t + (xk - 2.0 * q) / 6.0


def density_at_surface(s: ArrayLike, t: ArrayLike) -> np.ndarray:
    """
    Density of seawater at atmospheric pressure (Millero & Poisson 1981).

    The one-atmosphere International Equation of State of Seawater (IES 80)
    as given in UNESCO 1983.

    Check values: ``density_at_surface(0, 5)`` = 999.96675 kg/m³,
    ``density_at_surface(35, 5)`` = 1027.67547 kg/m³,
    ``density_at_surface(35, 25)`` = 1023.34306 kg/m³.

    :param s: Practical salinity (PSS-78).
    :type s: ArrayLike
    :param t: Temperature in °C (IPTS-68).
    :type t: ArrayLike
    :return: Density in kg/m³, NaN where an input is missing or outside the
             accepted domain.
    :rtype: numpy.ndarray
    """
    s = _in_domain(s, SALINITY_LIMITS)
    t = _in_domain(t, TEMPERATURE_LIMITS)

    # Density of Standard Mean Ocean Water (pure water, Bigg 1967).
    rho_w = (
        999.842594
        + 6.793952e-2 * t
        - 9.095290e-3 * t**2
        + 1.001685e-4 * t**3
        - 1.120083e-6 * t**4
        + 6.536332e-9 * t**5
    )

    b = (
        8.24493e-1
        - 4.0899e-3 * t
        + 7.6438e-5 * t**2
        - 8.2467e-7 * t**3
        + 5.3875e-9 * t**4
    )
    c = -5.72466e-3 + 1.0227e-4 * t - 1.6546e-6 * t**2
    d = 4.8314e-4

    return rho_w + b * s + c * s**1.5 + d * s**2


def sigma0(s: ArrayLike, t: ArrayLike, p: ArrayLike) -> np.ndarray:
    """
    Potential density anomaly sigma-0 of seawater.

    The density the water parcel would have at atmospheric pressure after
    an adiabatic move to the surface, minus 1000 kg/m³:
    ``rho(s, theta(s, t, p, 0), 0) - 1000``. This is the quantity compared
    at consecutive profile levels by the RTQC14 density inversion test.

    :param s: Practical salinity (PSS-78).
    :type s: ArrayLike
    :param t: In-situ temperature in °C (IPTS-68).
    :type t: ArrayLike
    :param p: Pressure in decibars.
    :type p: ArrayLike
    :return: Potential density anomaly in kg/m³, NaN where an input is
             missing or outside the accepted domain.
    :rtype: numpy.ndarray
    """
    theta = potential_temperature(s, t, p, p_ref=0.0)
    return density_at_surface(s, theta) - 1000.0


def gravity(lat: ArrayLike) -> np.ndarray:
    """
    Acceleration due to gravity at the sea surface (UNESCO 1983).

    The international gravity formula, varying with latitude because the
    Earth is neither spherical nor uniform.

    Check values: ``gravity(0)`` = 9.780318 m/s², ``gravity(90)`` =
    9.832177 m/s².

    :param lat: Latitude in degrees.
    :type lat: ArrayLike
    :return: Gravity in m/s², NaN where the latitude is missing or outside
             :data:`LATITUDE_LIMITS`.
    :rtype: numpy.ndarray
    """
    x = np.sin(np.radians(_in_domain(lat, LATITUDE_LIMITS))) ** 2
    return 9.780318 * (1.0 + (5.2788e-3 + 2.36e-5 * x) * x)


def depth_from_pressure(p: ArrayLike, lat: ArrayLike) -> np.ndarray:
    """
    Depth from pressure and latitude (UNESCO 1983).

    Converts a pressure measurement into metres below the surface,
    accounting for the latitude dependence of gravity and for the
    compression of the water column with depth.

    Check value: ``depth_from_pressure(10000, 30)`` = 9712.653 m.

    :param p: Pressure in decibars.
    :type p: ArrayLike
    :param lat: Latitude in degrees.
    :type lat: ArrayLike
    :return: Depth in metres, positive downward, NaN where an input is
             missing or outside the accepted domain.
    :rtype: numpy.ndarray
    """
    p = _in_domain(p, PRESSURE_LIMITS)
    # The local gravity used by the depth integral includes a small
    # correction for the pressure itself.
    g = gravity(lat) + 1.092e-6 * p
    column = (((-1.82e-15 * p + 2.279e-10) * p - 2.2512e-5) * p + 9.72659) * p
    return column / g


def brunt_vaisala_squared(
    sigma_theta: ArrayLike, d_sigma_d_depth: ArrayLike, lat: ArrayLike
) -> np.ndarray:
    """
    Brunt-Vaisala (buoyancy) frequency squared from a density gradient.

    ``N² = g / rho * d(sigma_theta)/dz`` with depth positive downward, so a
    water column whose potential density increases with depth is stable and
    gives a positive result, and an inversion gives a negative one. This is
    the "vertical density stability" of the feature proposal.

    The vertical gradient is an argument rather than something computed
    here: it is a property of the profile, not of the water parcel, and the
    profile is the caller's to walk (see
    :mod:`aiqclib.common.utils.profile_signal`). Keeping it out also keeps
    this module elementwise, so nothing here can reach across the boundary
    between two profiles.

    :param sigma_theta: Potential density anomaly in kg/m³, as returned by
                        :func:`sigma0`.
    :type sigma_theta: ArrayLike
    :param d_sigma_d_depth: Its vertical gradient in kg/m⁴, positive where
                            density increases downward.
    :type d_sigma_d_depth: ArrayLike
    :param lat: Latitude in degrees.
    :type lat: ArrayLike
    :return: N² in 1/s², NaN where an input is missing, the latitude is out
             of domain, or the implied density is not positive.
    :rtype: numpy.ndarray
    """
    rho = 1000.0 + _to_array(sigma_theta)
    rho = np.where(rho > 0.0, rho, np.nan)
    return gravity(lat) / rho * _to_array(d_sigma_d_depth)
