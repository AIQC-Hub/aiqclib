"""
Shared column-name constants used across the pipeline stages.

The observation-level pipeline carries all of :data:`ID_COLUMNS`; the
profile-level pipeline has no ``observation_no`` column, so step classes
must treat these lists as the superset of identity columns and tolerate
missing entries (see :func:`existing_columns` and polars'
``drop(..., strict=False)``).
"""

from typing import List

import polars as pl

#: Columns identifying a profile.
PROFILE_KEYS: List[str] = ["platform_code", "profile_no"]

#: Columns identifying a single observation across the pipeline.
OBSERVATION_KEYS: List[str] = PROFILE_KEYS + ["observation_no"]

#: Identity columns excluded from model features.
ID_COLUMNS: List[str] = ["row_id"] + OBSERVATION_KEYS

#: Identity columns plus the label, kept alongside predictions.
LABELED_ID_COLUMNS: List[str] = ID_COLUMNS + ["label"]


def existing_columns(df: pl.DataFrame, columns: List[str]) -> List[str]:
    """
    Return the subset of ``columns`` present in ``df``, in the given order.

    :param df: The DataFrame whose columns to check against.
    :type df: pl.DataFrame
    :param columns: Candidate column names.
    :type columns: List[str]
    :return: The names from ``columns`` that exist in ``df``.
    :rtype: List[str]
    """
    return [col for col in columns if col in df.columns]
