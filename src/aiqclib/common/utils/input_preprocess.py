"""
Automatic creation of the ``profile_no`` and ``observation_no`` identifier
columns.

Some raw inputs do not carry the sequential identifiers ``aiqclib`` needs.
When enabled in the configuration, this module derives them from other columns,
following the documented preprocessing recipe:

1. sort the rows so observations of one profile are grouped and ordered (by
   pressure);
2. build a temporary ``profile_key`` from the columns that together identify a
   profile (by default ``platform_code``, ``profile_timestamp``, ``longitude``
   and ``latitude``);
3. ``profile_no`` is the dense rank of that key within each ``platform_code``;
4. ``observation_no`` is the 1-indexed running count within each key;
5. the temporary key is dropped.

The set of columns to create, the key columns and the sort columns are all
configurable, so the inference can be tuned to a dataset (or disabled).

.. warning::
   The profile key must genuinely identify a profile. Slightly jittered
   coordinates would split one profile into several; identical timestamps at the
   same coordinates would merge distinct profiles. Choose the key columns
   accordingly.
"""

from typing import List, Optional

import polars as pl

from aiqclib.common.utils.input_validation import (
    FLOAT,
    INTEGER,
    REQUIRED_INPUT_COLUMNS,
)

#: Columns that, combined, identify a single profile.
DEFAULT_KEY_COLUMNS: List[str] = [
    "platform_code",
    "profile_timestamp",
    "longitude",
    "latitude",
]

#: Columns to sort by before numbering (the trailing ``pres`` orders
#: observations within a profile).
DEFAULT_SORT_COLUMNS: List[str] = [
    "platform_code",
    "profile_timestamp",
    "longitude",
    "latitude",
    "pres",
]

#: Identifier columns created by default.
DEFAULT_CREATED_COLUMNS: List[str] = ["profile_no", "observation_no"]

#: Column over which ``profile_no`` is ranked.
PLATFORM_COLUMN: str = "platform_code"

#: Internal temporary column name for the profile key.
_PROFILE_KEY: str = "__profile_key__"


def create_identifier_columns(
    df: pl.DataFrame,
    key_columns: Optional[List[str]] = None,
    sort_columns: Optional[List[str]] = None,
    columns: Optional[List[str]] = None,
    platform_column: str = PLATFORM_COLUMN,
) -> pl.DataFrame:
    """
    Create ``profile_no`` and/or ``observation_no`` from other columns.

    :param df: The input data (typically right after column renaming).
    :type df: pl.DataFrame
    :param key_columns: Columns whose combination identifies a profile.
                        Defaults to :data:`DEFAULT_KEY_COLUMNS`.
    :type key_columns: Optional[List[str]]
    :param sort_columns: Columns to sort by before numbering.
                         Defaults to :data:`DEFAULT_SORT_COLUMNS`.
    :type sort_columns: Optional[List[str]]
    :param columns: Which identifier columns to create; any subset of
                    ``["profile_no", "observation_no"]``. Defaults to both.
                    Listed columns are (re)generated, overwriting any existing
                    column of the same name.
    :type columns: Optional[List[str]]
    :param platform_column: Column over which ``profile_no`` is ranked.
    :type platform_column: str
    :raises ValueError: If a required source column is missing.
    :returns: The DataFrame with the requested identifier columns added.
    :rtype: pl.DataFrame
    """
    key_columns = list(key_columns or DEFAULT_KEY_COLUMNS)
    sort_columns = list(sort_columns or DEFAULT_SORT_COLUMNS)
    columns = list(columns or DEFAULT_CREATED_COLUMNS)

    create_profile_no = "profile_no" in columns
    create_observation_no = "observation_no" in columns
    if not (create_profile_no or create_observation_no):
        return df

    needed = set(key_columns) | set(sort_columns)
    if create_profile_no:
        needed.add(platform_column)
    missing = [name for name in needed if name not in df.columns]
    if missing:
        raise ValueError(
            "Cannot create identifier columns; the input is missing source "
            f"column(s): {', '.join(sorted(missing))}."
        )

    # Numeric sort columns read as strings (common with TSV/CSV) would sort
    # lexicographically (e.g. "10" before "2"), mis-ordering observations.
    # Cast such columns to numeric before sorting.
    coercions = [
        pl.col(name).cast(pl.Float64)
        for name in sort_columns
        if REQUIRED_INPUT_COLUMNS.get(name) in (FLOAT, INTEGER)
        and df.schema[name] == pl.Utf8
    ]
    if coercions:
        df = df.with_columns(coercions)

    df = df.sort(sort_columns)

    key_format = "|".join(["{}"] * len(key_columns))
    df = df.with_columns(
        pl.format(
            key_format, *[pl.col(name).cast(pl.Utf8) for name in key_columns]
        ).alias(_PROFILE_KEY)
    )

    created = []
    if create_profile_no:
        created.append(
            pl.col(_PROFILE_KEY)
            .rank("dense")
            .over(platform_column)
            .cast(pl.Int64)
            .alias("profile_no")
        )
    if create_observation_no:
        created.append(
            pl.col(_PROFILE_KEY)
            .cum_count()
            .over(_PROFILE_KEY)
            .cast(pl.Int64)
            .alias("observation_no")
        )

    return df.with_columns(created).drop(_PROFILE_KEY)
