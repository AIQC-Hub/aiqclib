"""
QC flag constants and aggregation helpers for the NRT QC module.

Defines the subset of the IOC/Argo flag scheme used by the NRT QC items
(1 = good, 3 = probably bad, 4 = bad) and a polars helper to aggregate
several per-item flag columns into the most severe flag per observation.
"""

from typing import Union

import polars as pl

#: A flag column referenced by name, or any polars expression yielding flags.
FlagExpr = Union[str, pl.Expr]

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
