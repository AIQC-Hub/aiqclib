"""
QC flag constants and helpers.

Defines the subset of the IOC/Argo flag scheme used by the NRT QC items
(1 = good, 3 = probably bad, 4 = bad), a polars helper to aggregate several
per-item flag columns into the most severe flag per observation, and the
helpers that read *existing* QC flag columns from input data.

Input sources disagree on how QC flags are stored: some write them as
integers, others as single-character strings (``"1"``, ``"4"``, ``""`` for
missing), and CSV/TSV inputs may arrive as floats. The configuration is
equally free to list flag values as ``[4, 6, 7]`` or ``["4", "6", "7"]``.
:func:`flag_as_int` and :func:`normalize_flag_values` reduce both sides to
Int64 so a comparison never depends on how the source spelled its flags.
"""

from typing import Iterable, List, Union

import polars as pl

#: A flag column referenced by name, or any polars expression yielding flags.
FlagExpr = Union[str, pl.Expr]

#: A configured flag value, as written in the YAML configuration.
FlagValue = Union[int, str, float]

#: Flag value for data that passed a QC item.
FLAG_GOOD: int = 1
#: Flag value for probably bad data (softened failure).
FLAG_PROBABLY_BAD: int = 3
#: Flag value for bad data (default failure).
FLAG_BAD: int = 4

#: Flag values ordered by ascending severity. Severity coincides with the
#: numeric order, which is what :func:`worst_flag` relies on.
FLAG_SEVERITY_ORDER: tuple = (FLAG_GOOD, FLAG_PROBABLY_BAD, FLAG_BAD)


def worst_flag(*flags: FlagExpr) -> pl.Expr:
    """
    Element-wise most severe flag across several flag columns/expressions.

    Because severity coincides with the numeric order for the flag scheme
    used here (1 < 3 < 4), the most severe flag is the horizontal maximum.
    Null entries are ignored, so a column that does not apply to a given
    observation cannot degrade the result.

    :param flags: Column names or polars expressions of flag values.
    :type flags: FlagExpr
    :return: An expression yielding the most severe flag per row.
    :rtype: polars.Expr
    """
    return pl.max_horizontal(*flags)


def flag_as_int(column: str) -> pl.Expr:
    """
    Read an existing QC flag column as Int64, whatever dtype it carries.

    Integer columns pass through, string columns are parsed, and float
    columns are truncated. Values that are not a number (``""``, ``NaN``,
    a non-numeric code) become null, so they match neither the positive nor
    the negative flag list and drop out of any selection.

    :param column: Name of the QC flag column in the input data.
    :type column: str
    :return: An expression yielding the flag values as Int64.
    :rtype: polars.Expr
    """
    # Float64 as the intermediate step parses "4" and "4.0" alike; both casts
    # are non-strict so unparseable entries become null instead of raising.
    return pl.col(column).cast(pl.Float64, strict=False).cast(pl.Int64, strict=False)


def normalize_flag_values(values: Iterable[FlagValue]) -> List[int]:
    """
    Convert configured flag values to the Int64 domain of :func:`flag_as_int`.

    Accepts the values as integers or as their string forms, so a
    configuration written for a string flag column keeps working against an
    integer one and vice versa.

    :param values: Flag values as written in the configuration.
    :type values: Iterable[FlagValue]
    :raises ValueError: If a value is not a whole number.
    :return: The values as integers.
    :rtype: List[int]
    """
    normalized: List[int] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"QC flag value {value!r} is not a number; flag values must be "
                f"whole numbers, written either as 4 or as '4'."
            ) from error
        if not number.is_integer():
            raise ValueError(
                f"QC flag value {value!r} is not a whole number; flag values "
                f"must be whole numbers, written either as 4 or as '4'."
            )
        normalized.append(int(number))
    return normalized


def flag_is_in(column: str, values: Iterable[FlagValue]) -> pl.Expr:
    """
    Test membership of a QC flag column in a configured list of flag values.

    Both sides are normalized first, so the test works for any combination of
    flag column dtype and configured value type. A flag that :func:`flag_as_int`
    could not read is not a match: the result is ``False`` rather than null, so
    the outcome does not depend on how a caller handles null propagation.

    :param column: Name of the QC flag column in the input data.
    :type column: str
    :param values: The flag values to match, as written in the configuration.
    :type values: Iterable[FlagValue]
    :return: A boolean expression, false for unparseable flags.
    :rtype: polars.Expr
    """
    return flag_as_int(column).is_in(normalize_flag_values(values)).fill_null(False)
