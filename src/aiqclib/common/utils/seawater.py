"""
EOS-80 seawater routines (UNESCO 1983) for the NRT QC module.

Implements the equation of state of seawater from Fofonoff & Millard (1983),
UNESCO Technical Papers in Marine Science #44: the adiabatic lapse rate,
potential temperature, density at atmospheric pressure, and the potential
density anomaly sigma-0 used by the RTQC14 density inversion test.

All functions follow the UNESCO argument order (salinity, temperature,
pressure) with practical salinity (PSS-78), in-situ temperature in degrees
Celsius (IPTS-68), and pressure in decibars. Inputs may be scalars, numpy
arrays, or polars Series; computation is vectorised with numpy and NaN
values propagate through the results.
"""

from typing import Union

import numpy as np
import polars as pl

ArrayLike = Union[float, list, np.ndarray, pl.Series]


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
    :return: Adiabatic lapse rate in °C/dbar.
    :rtype: numpy.ndarray
    """
    s = _to_array(s)
    t = _to_array(t)
    p = _to_array(p)

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
    :return: Potential temperature in °C referenced to ``p_ref``.
    :rtype: numpy.ndarray
    """
    s = _to_array(s)
    t = np.array(_to_array(t), copy=True)
    p = np.array(_to_array(p), copy=True)

    h = p_ref - p
    xk = h * adiabatic_lapse_rate(s, t, p)
    t = t + 0.5 * xk
    q = xk
    p = p + 0.5 * h
    xk = h * adiabatic_lapse_rate(s, t, p)
    t = t + 0.29289322 * (xk - q)
    q = 0.58578644 * xk + 0.121320344 * q
    xk = h * adiabatic_lapse_rate(s, t, p)
    t = t + 1.707106781 * (xk - q)
    q = 3.414213562 * xk - 4.121320344 * q
    p = p + 0.5 * h
    xk = h * adiabatic_lapse_rate(s, t, p)
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
    :return: Density in kg/m³.
    :rtype: numpy.ndarray
    """
    s = _to_array(s)
    t = _to_array(t)

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
    :return: Potential density anomaly in kg/m³.
    :rtype: numpy.ndarray
    """
    theta = potential_temperature(s, t, p, p_ref=0.0)
    return density_at_surface(s, theta) - 1000.0
