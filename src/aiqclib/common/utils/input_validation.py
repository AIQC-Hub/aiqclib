"""
Validation and automatic type correction for mandatory input columns.

``aiqclib`` requires every input dataset to provide a small set of identity and
coordinate columns with specific data types. This module centralises:

- :data:`REQUIRED_INPUT_COLUMNS`, the editable table of mandatory columns and
  their expected logical types; and
- :func:`validate_and_convert_input_columns`, which checks that those columns
  are present and, where a column has the wrong type, attempts to convert it.

The validation is intended to run immediately after column renaming, so it sees
the final column names. Automatic conversion is especially useful for TSV/CSV
inputs, where numeric and datetime columns are frequently read as strings. As
noted below, datetime conversion can only be done automatically for genuine
date/datetime values (or string representations of them); numeric epoch encodings
are ambiguous and must be converted up front (see the data-preprocessing guide).
"""

from typing import Dict, List, Optional

import polars as pl

# --------------------------------------------------------------------------- #
# Logical type categories.
# --------------------------------------------------------------------------- #
TEXT = "text"
INTEGER = "integer"
FLOAT = "float"
DATETIME = "datetime"

# --------------------------------------------------------------------------- #
# The editable table of mandatory input columns and their expected types.
#
# Edit this mapping to change which columns are required or what type each is
# expected to have. Keys are column names (after renaming); values are one of
# the logical type categories above.
# --------------------------------------------------------------------------- #
REQUIRED_INPUT_COLUMNS: Dict[str, str] = {
    "platform_code": TEXT,
    "profile_no": INTEGER,
    "profile_timestamp": DATETIME,
    "longitude": FLOAT,
    "latitude": FLOAT,
    "observation_no": INTEGER,
    "pres": FLOAT,
}

# Canonical target dtype used when casting a column to a given category.
_TARGET_DTYPE = {
    TEXT: pl.Utf8,
    INTEGER: pl.Int64,
    FLOAT: pl.Float64,
    DATETIME: pl.Datetime("ms"),
}

# Acceptable existing dtypes per category (no conversion needed when matched).
_INTEGER_DTYPES = (
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
)
_FLOAT_DTYPES = (pl.Float32, pl.Float64)


def _matches_category(dtype: pl.DataType, category: str) -> bool:
    """
    Return whether an existing Polars dtype already satisfies a category.

    :param dtype: The column's current Polars dtype.
    :type dtype: pl.DataType
    :param category: One of :data:`TEXT`, :data:`INTEGER`, :data:`FLOAT`,
                     :data:`DATETIME`.
    :type category: str
    :returns: ``True`` if no conversion is needed.
    :rtype: bool
    """
    if category == TEXT:
        return dtype == pl.Utf8
    if category == INTEGER:
        return dtype in _INTEGER_DTYPES
    if category == FLOAT:
        return dtype in _FLOAT_DTYPES
    if category == DATETIME:
        return isinstance(dtype, pl.Datetime)
    raise ValueError(f"Unknown column type category: '{category}'.")


def _conversion_expr(name: str, dtype: pl.DataType, category: str) -> pl.Expr:
    """
    Build a Polars expression that converts a column to the expected category.

    :param name: Column name.
    :type name: str
    :param dtype: The column's current dtype.
    :type dtype: pl.DataType
    :param category: The expected logical type category.
    :type category: str
    :raises ValueError: If a datetime cannot be derived automatically (e.g. the
                        column is a numeric epoch encoding, which is ambiguous).
    :returns: A Polars expression producing the converted column.
    :rtype: pl.Expr
    """
    if category == TEXT:
        return pl.col(name).cast(pl.Utf8)

    if category == FLOAT:
        return pl.col(name).cast(pl.Float64)

    if category == INTEGER:
        # Route string columns through Float so values like "1" and "1.0" both
        # convert cleanly; other dtypes cast directly.
        if dtype == pl.Utf8:
            return pl.col(name).cast(pl.Float64).cast(pl.Int64)
        return pl.col(name).cast(pl.Int64)

    if category == DATETIME:
        if isinstance(dtype, pl.Datetime):
            return pl.col(name)
        if dtype == pl.Date:
            return pl.col(name).cast(_TARGET_DTYPE[DATETIME])
        if dtype == pl.Utf8:
            # Infer the format from the string representation.
            return pl.col(name).str.to_datetime()
        raise ValueError(
            f"Column '{name}' has type {dtype}, which cannot be converted to a "
            f"datetime automatically. Numeric timestamps (e.g. days since an "
            f"epoch) are ambiguous; convert '{name}' to 'profile_timestamp' "
            f"before running aiqclib (see the data-preprocessing guide)."
        )

    raise ValueError(f"Unknown column type category: '{category}'.")


def validate_and_convert_input_columns(
    df: pl.DataFrame,
    required_columns: Optional[Dict[str, str]] = None,
) -> pl.DataFrame:
    """
    Validate mandatory input columns and convert mismatched types in place.

    For each entry in ``required_columns`` this checks that the column exists
    and that its dtype matches the expected category. Columns with the wrong
    type are converted where possible (e.g. numeric strings from CSV/TSV become
    floats/integers, and date/datetime strings become datetimes).

    :param df: The input data, typically immediately after column renaming.
    :type df: pl.DataFrame
    :param required_columns: The mandatory-column table to validate against.
                             Defaults to :data:`REQUIRED_INPUT_COLUMNS`.
    :type required_columns: Optional[Dict[str, str]]
    :raises ValueError: If any required column is missing, or if a column's type
                        cannot be converted to the expected type.
    :returns: The validated DataFrame, with any necessary conversions applied.
    :rtype: pl.DataFrame
    """
    required_columns = required_columns or REQUIRED_INPUT_COLUMNS

    missing = [name for name in required_columns if name not in df.columns]
    if missing:
        raise ValueError(
            "Input data is missing required column(s): "
            f"{', '.join(missing)}. Expected columns: "
            f"{', '.join(required_columns)}."
        )

    for name, category in required_columns.items():
        current_dtype = df.schema[name]
        if _matches_category(current_dtype, category):
            continue

        expr = _conversion_expr(name, current_dtype, category)
        try:
            df = df.with_columns(expr)
        except Exception as error:  # noqa: BLE001 - re-raised with context
            raise ValueError(
                f"Could not convert column '{name}' from {current_dtype} to "
                f"the expected '{category}' type: {error}"
            ) from error

        if not _matches_category(df.schema[name], category):
            raise ValueError(
                f"Column '{name}' is {df.schema[name]} after conversion but "
                f"the expected type is '{category}'."
            )

    return df


def required_column_names() -> List[str]:
    """
    Return the list of mandatory input column names.

    :returns: Names from :data:`REQUIRED_INPUT_COLUMNS`, in definition order.
    :rtype: List[str]
    """
    return list(REQUIRED_INPUT_COLUMNS)
