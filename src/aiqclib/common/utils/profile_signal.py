"""
Vertical signal processing over CTD profiles, as polars expressions.

The feature classes of the point-level, window-level and stratification
groups all need the same handful of operations along a profile: smooth it,
differentiate it, compare a level with its neighbours, and describe the
window around a level robustly. This module holds those operations once.

Two rules shape the API:

- **Every operation is partitioned by** :data:`~aiqclib.common.constants.PROFILE_KEYS`
  and ordered by ``observation_no``, so a value never reaches across the
  boundary between two profiles. The ordering is passed to polars rather
  than assumed, so a caller cannot get a silently wrong answer by handing
  over an unsorted frame.
- **Profile edges are null.** A centred window of width ``w`` needs
  ``(w - 1) / 2`` levels on each side; where they do not exist the result is
  null rather than a reflected, extended or shrunken estimate. Null is what
  the models already treat as missing, and it cannot be mistaken for a
  measurement.

Most operations are plain expression builders. The few that polars cannot
express as a single window function (anything built on a median of
deviations from a median) are DataFrame helpers instead, named
``with_*``, that add their column and clean up their intermediates.
"""

from functools import lru_cache
from math import factorial
from typing import List, Optional, Sequence, Tuple

import numpy as np
import polars as pl

from aiqclib.common.constants import PROFILE_KEYS

#: Consistency factor making the median absolute deviation comparable with
#: the standard deviation for normally distributed data.
MAD_SCALE: float = 1.4826

#: The rolling statistics :func:`rolling_stat_expr` can build.
ROLLING_STATS: Tuple[str, ...] = ("mean", "median", "min", "max", "std", "sum")

#: The directions :func:`neighbor_diff_expr` understands. ``"up"`` compares
#: a level with a shallower one (earlier in the profile), ``"down"`` with a
#: deeper one.
DIFF_DIRECTIONS: Tuple[str, ...] = ("up", "down")


def _over(
    expr: pl.Expr,
    keys: Sequence[str],
    order_by: Optional[str],
) -> pl.Expr:
    """
    Evaluate an expression within each profile, in observation order.

    :param expr: The expression to partition.
    :type expr: pl.Expr
    :param keys: The columns identifying a profile.
    :type keys: Sequence[str]
    :param order_by: The column giving the order within a profile, or
                     :obj:`None` when the frame is already sorted.
    :type order_by: Optional[str]
    :return: The expression wrapped in a window function.
    :rtype: pl.Expr
    """
    if order_by is None:
        return expr.over(list(keys))
    return expr.over(list(keys), order_by=order_by)


def sort_profiles(
    df: pl.DataFrame,
    keys: Sequence[str] = PROFILE_KEYS,
    order_column: str = "observation_no",
) -> pl.DataFrame:
    """
    Sort a frame into profile order.

    The expression builders order within the window themselves, so this is
    not needed for correctness; it is here because a sorted frame makes the
    intermediate results of a feature class readable when debugging.

    :param df: The frame to sort.
    :type df: pl.DataFrame
    :param keys: The columns identifying a profile.
    :type keys: Sequence[str]
    :param order_column: The column giving the order within a profile.
    :type order_column: str
    :return: The sorted frame.
    :rtype: pl.DataFrame
    """
    return df.sort([*keys, order_column])


# --------------------------------------------------------------------------- #
# Savitzky-Golay smoothing and derivatives.
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=64)
def savgol_coefficients(
    window: int, polyorder: int, deriv: int = 0
) -> Tuple[float, ...]:
    """
    Coefficients of the centred Savitzky-Golay filter.

    A Savitzky-Golay filter fits a polynomial of degree ``polyorder`` to the
    ``window`` points around a level by least squares and evaluates that
    polynomial, or one of its derivatives, at the centre. Because the fit is
    linear in the data and the sample positions are fixed, the whole
    operation collapses to one fixed set of weights, which is what this
    returns. That is why no signal processing dependency is needed: the
    filter is a weighted sum of shifted columns (see :func:`fir_expr`).

    Derivatives are with respect to the sample index, not to depth. A caller
    wanting a per-decibar gradient divides by the local sample spacing.

    :param window: The number of points in the window. Must be odd and at
                   least ``polyorder + 1``.
    :type window: int
    :param polyorder: The degree of the fitted polynomial.
    :type polyorder: int
    :param deriv: The order of the derivative to evaluate, 0 for the
                  smoothed value itself.
    :type deriv: int
    :return: ``window`` weights, ordered from the shallowest point of the
             window to the deepest.
    :rtype: Tuple[float, ...]
    :raises ValueError: If the window is even, too short, or the derivative
                        order exceeds the polynomial degree.
    """
    if window < 3 or window % 2 == 0:
        raise ValueError(f"Savitzky-Golay window must be odd and >= 3, got {window}.")
    if polyorder < 0 or polyorder >= window:
        raise ValueError(
            f"Savitzky-Golay polyorder must be in [0, {window - 1}], got {polyorder}."
        )
    if deriv < 0 or deriv > polyorder:
        raise ValueError(
            f"Savitzky-Golay derivative order must be in [0, {polyorder}], got {deriv}."
        )

    half = (window - 1) // 2
    positions = np.arange(-half, half + 1, dtype=np.float64)
    design = np.vander(positions, polyorder + 1, increasing=True)
    # Row `deriv` of the pseudo-inverse gives the polynomial coefficient of
    # that power; the derivative at the centre is that coefficient times the
    # factorial of the order.
    weights = factorial(deriv) * np.linalg.pinv(design)[deriv]
    return tuple(float(value) for value in weights)


def fir_expr(
    column: str,
    coefficients: Sequence[float],
    keys: Sequence[str] = PROFILE_KEYS,
    order_by: Optional[str] = "observation_no",
) -> pl.Expr:
    """
    Apply a centred finite impulse response filter along each profile.

    The output at a level is the weighted sum of the values around it, with
    ``coefficients`` ordered from the shallowest point of the window to the
    deepest. Levels without a full window are null.

    :param column: The column to filter.
    :type column: str
    :param coefficients: The filter weights; the length must be odd.
    :type coefficients: Sequence[float]
    :param keys: The columns identifying a profile.
    :type keys: Sequence[str]
    :param order_by: The column giving the order within a profile.
    :type order_by: Optional[str]
    :return: The filtered column as an expression.
    :rtype: pl.Expr
    :raises ValueError: If the number of coefficients is even.
    """
    if len(coefficients) % 2 == 0:
        raise ValueError(
            f"A centred filter needs an odd number of coefficients, "
            f"got {len(coefficients)}."
        )

    half = (len(coefficients) - 1) // 2
    terms = [
        pl.col(column).shift(half - offset) * float(weight)
        for offset, weight in enumerate(coefficients)
        if weight != 0.0
    ]
    total = terms[0]
    for term in terms[1:]:
        total = total + term
    return _over(total, keys, order_by)


def savgol_expr(
    column: str,
    window: int,
    polyorder: int = 2,
    deriv: int = 0,
    keys: Sequence[str] = PROFILE_KEYS,
    order_by: Optional[str] = "observation_no",
) -> pl.Expr:
    """
    The Savitzky-Golay smoothed value, or one of its derivatives.

    :param column: The column to smooth.
    :type column: str
    :param window: The number of points in the window (odd).
    :type window: int
    :param polyorder: The degree of the fitted polynomial.
    :type polyorder: int
    :param deriv: The derivative order, 0 for the smoothed value.
    :type deriv: int
    :param keys: The columns identifying a profile.
    :type keys: Sequence[str]
    :param order_by: The column giving the order within a profile.
    :type order_by: Optional[str]
    :return: The filtered column as an expression, per sample index.
    :rtype: pl.Expr
    """
    return fir_expr(
        column,
        savgol_coefficients(window, polyorder, deriv),
        keys=keys,
        order_by=order_by,
    )


# --------------------------------------------------------------------------- #
# Neighbour comparisons.
# --------------------------------------------------------------------------- #
def neighbor_diff_expr(
    column: str,
    lag: int,
    direction: str = "up",
    keys: Sequence[str] = PROFILE_KEYS,
    order_by: Optional[str] = "observation_no",
) -> pl.Expr:
    """
    The difference between a level and one of its vertical neighbours.

    ``"up"`` subtracts the value ``lag`` levels shallower, ``"down"`` the
    value ``lag`` levels deeper. The first ``lag`` levels of a profile have
    no upward neighbour and the last ``lag`` no downward one, so those are
    null.

    :param column: The column to difference.
    :type column: str
    :param lag: How many levels away the neighbour is.
    :type lag: int
    :param direction: ``"up"`` or ``"down"``.
    :type direction: str
    :param keys: The columns identifying a profile.
    :type keys: Sequence[str]
    :param order_by: The column giving the order within a profile.
    :type order_by: Optional[str]
    :return: The difference as an expression.
    :rtype: pl.Expr
    :raises ValueError: If the lag is not positive or the direction unknown.
    """
    if lag < 1:
        raise ValueError(f"Neighbour lag must be at least 1, got {lag}.")
    if direction not in DIFF_DIRECTIONS:
        raise ValueError(
            f"Unknown neighbour direction '{direction}'. "
            f"Valid directions: {list(DIFF_DIRECTIONS)}."
        )

    shift = lag if direction == "up" else -lag
    return _over(pl.col(column) - pl.col(column).shift(shift), keys, order_by)


def central_difference_expr(
    value_column: str,
    coordinate_column: str,
    keys: Sequence[str] = PROFILE_KEYS,
    order_by: Optional[str] = "observation_no",
    min_separation: float = 0.0,
) -> pl.Expr:
    """
    The rate of change of one column with respect to another.

    A centred difference across the two neighbouring levels, which is the
    usual way to read a vertical gradient off a profile: it is second order
    accurate and, unlike a one-sided difference, does not shift the feature
    it measures half a level up or down.

    The first and last level of a profile have no pair of neighbours and are
    null. So is any level whose neighbours are separated by no more than
    ``min_separation``, since dividing by a separation of nearly zero turns
    a rounding difference into an enormous gradient.

    :param value_column: The column being differentiated.
    :type value_column: str
    :param coordinate_column: The column differentiated against, such as
                              depth or pressure.
    :type coordinate_column: str
    :param keys: The columns identifying a profile.
    :type keys: Sequence[str]
    :param order_by: The column giving the order within a profile.
    :type order_by: Optional[str]
    :param min_separation: The smallest coordinate separation to divide by.
    :type min_separation: float
    :return: The gradient as an expression.
    :rtype: pl.Expr
    """
    d_value = pl.col(value_column).shift(-1) - pl.col(value_column).shift(1)
    d_coordinate = pl.col(coordinate_column).shift(-1) - pl.col(
        coordinate_column
    ).shift(1)
    gradient = (
        pl.when(d_coordinate.abs() > min_separation)
        .then(d_value / d_coordinate)
        .otherwise(None)
    )
    return _over(gradient, keys, order_by)


def spike_index_expr(v1: pl.Expr, v2: pl.Expr, v3: pl.Expr) -> pl.Expr:
    """
    The Argo spike test value for a level and its two neighbours.

    ``|V2 - (V3 + V1) / 2| - |(V3 - V1) / 2|``: how far the level sits from
    the average of its neighbours, less the size of the gradient across
    them, so that a genuine gradient does not score as a spike. This is the
    quantity RTQC9 thresholds; as a feature it is used unthresholded.

    :param v1: The value at the level above.
    :type v1: pl.Expr
    :param v2: The value at the level itself.
    :type v2: pl.Expr
    :param v3: The value at the level below.
    :type v3: pl.Expr
    :return: The spike test value as an expression.
    :rtype: pl.Expr
    """
    return (v2 - (v3 + v1) / 2).abs() - ((v3 - v1) / 2).abs()


def spike_index_column_expr(
    column: str,
    keys: Sequence[str] = PROFILE_KEYS,
    order_by: Optional[str] = "observation_no",
) -> pl.Expr:
    """
    The spike test value of a column against its immediate neighbours.

    The first and last level of a profile have no pair of neighbours, so
    they are null.

    :param column: The column to score.
    :type column: str
    :param keys: The columns identifying a profile.
    :type keys: Sequence[str]
    :param order_by: The column giving the order within a profile.
    :type order_by: Optional[str]
    :return: The spike test value as an expression.
    :rtype: pl.Expr
    """
    return _over(
        spike_index_expr(
            pl.col(column).shift(1), pl.col(column), pl.col(column).shift(-1)
        ),
        keys,
        order_by,
    )


# --------------------------------------------------------------------------- #
# Window statistics.
# --------------------------------------------------------------------------- #
def rolling_stat_expr(
    column: str,
    window: int,
    stat: str,
    keys: Sequence[str] = PROFILE_KEYS,
    order_by: Optional[str] = "observation_no",
    min_samples: Optional[int] = None,
) -> pl.Expr:
    """
    A centred rolling statistic along each profile.

    :param column: The column to summarise.
    :type column: str
    :param window: The number of points in the window (odd).
    :type window: int
    :param stat: One of :data:`ROLLING_STATS`.
    :type stat: str
    :param keys: The columns identifying a profile.
    :type keys: Sequence[str]
    :param order_by: The column giving the order within a profile.
    :type order_by: Optional[str]
    :param min_samples: How many non-null values the window needs; defaults
                        to the full window, which makes profile edges null.
    :type min_samples: Optional[int]
    :return: The rolling statistic as an expression.
    :rtype: pl.Expr
    :raises ValueError: If the statistic is unknown or the window even.
    """
    if stat not in ROLLING_STATS:
        raise ValueError(
            f"Unknown rolling statistic '{stat}'. Valid statistics: "
            f"{list(ROLLING_STATS)}."
        )
    if window < 1 or window % 2 == 0:
        raise ValueError(f"A centred window must be odd and >= 1, got {window}.")

    rolling = getattr(pl.col(column), f"rolling_{stat}")
    return _over(
        rolling(
            window_size=window,
            center=True,
            min_samples=window if min_samples is None else min_samples,
        ),
        keys,
        order_by,
    )


def fraction_expr(
    condition: pl.Expr,
    window: int,
    keys: Sequence[str] = PROFILE_KEYS,
    order_by: Optional[str] = "observation_no",
    min_samples: int = 1,
) -> pl.Expr:
    """
    The fraction of the window around a level that satisfies a condition.

    Used for the window-level percentages of the feature proposal: outliers
    in the residual, large neighbour differences, high curvature. Unlike the
    other window statistics this defaults to ``min_samples=1``, because the
    quantities being counted are themselves null at profile edges and a
    fraction over the levels that do exist is more useful than no fraction
    at all.

    :param condition: A boolean expression over columns of the frame. It
                      must not itself be a window function.
    :type condition: pl.Expr
    :param window: The number of points in the window (odd).
    :type window: int
    :param keys: The columns identifying a profile.
    :type keys: Sequence[str]
    :param order_by: The column giving the order within a profile.
    :type order_by: Optional[str]
    :param min_samples: How many non-null values the window needs.
    :type min_samples: int
    :return: The fraction in ``[0, 1]`` as an expression.
    :rtype: pl.Expr
    :raises ValueError: If the window is even.
    """
    if window < 1 or window % 2 == 0:
        raise ValueError(f"A centred window must be odd and >= 1, got {window}.")

    return _over(
        condition.cast(pl.Float64).rolling_mean(
            window_size=window, center=True, min_samples=min_samples
        ),
        keys,
        order_by,
    )


# --------------------------------------------------------------------------- #
# Robust statistics.
#
# A median absolute deviation is a median of deviations from a median, which
# polars cannot express as a single window function (one window function
# cannot be nested inside another). These helpers add the column in stages
# and drop their intermediates.
# --------------------------------------------------------------------------- #
def _robust_z_expr(value: pl.Expr, center: pl.Expr, deviation: pl.Expr) -> pl.Expr:
    """
    Standardise a value by a median and a median absolute deviation.

    The result is null where the deviation is zero: a window whose values
    are all identical gives no scale to measure an anomaly against, and
    calling that anomaly infinite would be a claim the data does not make.

    :param value: The value to standardise.
    :type value: pl.Expr
    :param center: The median to subtract.
    :type center: pl.Expr
    :param deviation: The median absolute deviation to divide by.
    :type deviation: pl.Expr
    :return: The robust z-score as an expression.
    :rtype: pl.Expr
    """
    scale = MAD_SCALE * deviation
    return pl.when(scale > 0).then((value - center) / scale).otherwise(None)


def _with_temporaries(
    df: pl.DataFrame, stages: List[pl.Expr], temporaries: List[str]
) -> pl.DataFrame:
    """
    Apply expression stages in order, then drop the intermediate columns.

    :param df: The frame to extend.
    :type df: pl.DataFrame
    :param stages: One expression per stage, applied in order.
    :type stages: List[pl.Expr]
    :param temporaries: The intermediate column names to drop at the end.
    :type temporaries: List[str]
    :return: The frame with the final column added and intermediates gone.
    :rtype: pl.DataFrame
    """
    for stage in stages:
        df = df.with_columns(stage)
    return df.drop(temporaries)


def window_list_expr(
    column: str,
    window: int,
    keys: Sequence[str] = PROFILE_KEYS,
    order_by: Optional[str] = "observation_no",
) -> pl.Expr:
    """
    Gather the centred window around each level into a list column.

    The window statistics polars provides directly cannot express a median
    absolute deviation, because that is a median of deviations from *the
    window's own* median: a rolling median of a column of local residuals
    is a different quantity, and a plausible looking one, which is exactly
    why it is worth naming. Materialising the window makes the definition
    literal, at the cost of one list column.

    :param column: The column to gather.
    :type column: str
    :param window: The number of points in the window (odd).
    :type window: int
    :param keys: The columns identifying a profile.
    :type keys: Sequence[str]
    :param order_by: The column giving the order within a profile.
    :type order_by: Optional[str]
    :return: A list column holding each level's window, shallowest first,
             with nulls where the window runs past the profile.
    :rtype: pl.Expr
    :raises ValueError: If the window is even.
    """
    if window < 1 or window % 2 == 0:
        raise ValueError(f"A centred window must be odd and >= 1, got {window}.")

    half = (window - 1) // 2
    return _over(
        pl.concat_list(
            [pl.col(column).shift(half - offset) for offset in range(window)]
        ),
        keys,
        order_by,
    )


def _enough_samples(window_column: str, min_samples: int) -> pl.Expr:
    """
    Whether a materialised window holds enough non-null values to use.

    :param window_column: The list column holding the window.
    :type window_column: str
    :param min_samples: The number of non-null values required.
    :type min_samples: int
    :return: A boolean expression.
    :rtype: pl.Expr
    """
    return pl.col(window_column).list.drop_nulls().list.len() >= min_samples


def with_rolling_mad(
    df: pl.DataFrame,
    column: str,
    window: int,
    alias: str,
    keys: Sequence[str] = PROFILE_KEYS,
    order_by: Optional[str] = "observation_no",
    min_samples: Optional[int] = None,
) -> pl.DataFrame:
    """
    Add the centred rolling median absolute deviation of a column.

    The median of ``|x - median(window)|`` over the window, which is the
    scale a robust z-score divides by.

    :param df: The frame to extend.
    :type df: pl.DataFrame
    :param column: The column to summarise.
    :type column: str
    :param window: The number of points in the window (odd).
    :type window: int
    :param alias: The name of the column to add.
    :type alias: str
    :param keys: The columns identifying a profile.
    :type keys: Sequence[str]
    :param order_by: The column giving the order within a profile.
    :type order_by: Optional[str]
    :param min_samples: How many non-null values the window needs; defaults
                        to the full window.
    :type min_samples: Optional[int]
    :return: The frame with ``alias`` added.
    :rtype: pl.DataFrame
    """
    values = f"_{alias}_window"
    required = window if min_samples is None else min_samples
    return _with_temporaries(
        df,
        [
            window_list_expr(column, window, keys, order_by).alias(values),
            pl.when(_enough_samples(values, required))
            .then(
                pl.col(values)
                .list.eval((pl.element() - pl.element().median()).abs().median())
                .list.first()
            )
            .otherwise(None)
            .alias(alias),
        ],
        [values],
    )


def with_rolling_robust_z(
    df: pl.DataFrame,
    column: str,
    window: int,
    alias: str,
    keys: Sequence[str] = PROFILE_KEYS,
    order_by: Optional[str] = "observation_no",
    min_samples: Optional[int] = None,
) -> pl.DataFrame:
    """
    Add the robust z-score of a column against its own centred window.

    :param df: The frame to extend.
    :type df: pl.DataFrame
    :param column: The column to standardise.
    :type column: str
    :param window: The number of points in the window (odd).
    :type window: int
    :param alias: The name of the column to add.
    :type alias: str
    :param keys: The columns identifying a profile.
    :type keys: Sequence[str]
    :param order_by: The column giving the order within a profile.
    :type order_by: Optional[str]
    :param min_samples: How many non-null values the window needs; defaults
                        to the full window.
    :type min_samples: Optional[int]
    :return: The frame with ``alias`` added.
    :rtype: pl.DataFrame
    """
    values = f"_{alias}_window"
    median = f"_{alias}_median"
    mad = f"_{alias}_mad"
    required = window if min_samples is None else min_samples
    return _with_temporaries(
        df,
        [
            window_list_expr(column, window, keys, order_by).alias(values),
            pl.when(_enough_samples(values, required))
            .then(pl.col(values).list.median())
            .otherwise(None)
            .alias(median),
            pl.when(_enough_samples(values, required))
            .then(
                pl.col(values)
                .list.eval((pl.element() - pl.element().median()).abs().median())
                .list.first()
            )
            .otherwise(None)
            .alias(mad),
            _robust_z_expr(pl.col(column), pl.col(median), pl.col(mad)).alias(alias),
        ],
        [values, median, mad],
    )


def with_profile_mad(
    df: pl.DataFrame,
    column: str,
    alias: str,
    keys: Sequence[str] = PROFILE_KEYS,
) -> pl.DataFrame:
    """
    Add the median absolute deviation of a column over the whole profile.

    :param df: The frame to extend.
    :type df: pl.DataFrame
    :param column: The column to summarise.
    :type column: str
    :param alias: The name of the column to add.
    :type alias: str
    :param keys: The columns identifying a profile.
    :type keys: Sequence[str]
    :return: The frame with ``alias`` added, constant within each profile.
    :rtype: pl.DataFrame
    """
    median = f"_{alias}_median"
    deviation = f"_{alias}_deviation"
    partition = list(keys)
    return _with_temporaries(
        df,
        [
            pl.col(column).median().over(partition).alias(median),
            (pl.col(column) - pl.col(median)).abs().alias(deviation),
            pl.col(deviation).median().over(partition).alias(alias),
        ],
        [median, deviation],
    )


def with_profile_robust_z(
    df: pl.DataFrame,
    column: str,
    alias: str,
    keys: Sequence[str] = PROFILE_KEYS,
) -> pl.DataFrame:
    """
    Add the robust z-score of a column against its whole profile.

    :param df: The frame to extend.
    :type df: pl.DataFrame
    :param column: The column to standardise.
    :type column: str
    :param alias: The name of the column to add.
    :type alias: str
    :param keys: The columns identifying a profile.
    :type keys: Sequence[str]
    :return: The frame with ``alias`` added.
    :rtype: pl.DataFrame
    """
    median = f"_{alias}_median"
    deviation = f"_{alias}_deviation"
    mad = f"_{alias}_mad"
    partition = list(keys)
    return _with_temporaries(
        df,
        [
            pl.col(column).median().over(partition).alias(median),
            (pl.col(column) - pl.col(median)).abs().alias(deviation),
            pl.col(deviation).median().over(partition).alias(mad),
            _robust_z_expr(pl.col(column), pl.col(median), pl.col(mad)).alias(alias),
        ],
        [median, deviation, mad],
    )
