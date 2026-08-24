"""
Per-profile aggregation expressions for the profile-level pipeline.

Observation-level features are turned into profile-level features by
aggregating each feature column over the observations of a profile. The
aggregations are requested per feature via the ``agg`` key of a
``feature_param_sets`` entry and resolved here into polars expressions.
"""

from typing import Callable, Dict, List

import polars as pl

from aiqclib.common.utils.qc_flags import FLAG_GOOD

#: Mapping of aggregation names to expression builders. Each builder takes
#: a column name and returns an aggregation expression aliased to
#: ``{column}_{agg}``.
AGGREGATION_REGISTRY: Dict[str, Callable[[str], pl.Expr]] = {
    "mean": lambda col: pl.col(col).mean().alias(f"{col}_mean"),
    "min": lambda col: pl.col(col).min().alias(f"{col}_min"),
    "max": lambda col: pl.col(col).max().alias(f"{col}_max"),
    "median": lambda col: pl.col(col).median().alias(f"{col}_median"),
    "std": lambda col: pl.col(col).std().alias(f"{col}_std"),
    "sum": lambda col: pl.col(col).sum().alias(f"{col}_sum"),
    "first": lambda col: pl.col(col).first().alias(f"{col}_first"),
    # QC-flag oriented aggregations: the fraction of observations failing
    # the item, and whether any observation fails it.
    "fail_frac": lambda col: (
        (pl.col(col) != FLAG_GOOD).mean().alias(f"{col}_fail_frac")
    ),
    "fail_any": lambda col: (
        (pl.col(col) != FLAG_GOOD).any().cast(pl.Int8).alias(f"{col}_fail_any")
    ),
}


def validate_aggregation_names(feature_name: str, agg_names: List[str]) -> None:
    """
    Raise when a requested aggregation name is unknown.

    :param feature_name: The feature the aggregations were requested for,
                         used in the error message.
    :type feature_name: str
    :param agg_names: The aggregation names from the feature's ``agg`` key.
    :type agg_names: List[str]
    :raises ValueError: If a name is not in :data:`AGGREGATION_REGISTRY`.
    """
    unknown = [name for name in agg_names if name not in AGGREGATION_REGISTRY]
    if unknown:
        raise ValueError(
            f"Unknown aggregation(s) {unknown} for feature '{feature_name}'. "
            f"Valid aggregations: {sorted(AGGREGATION_REGISTRY)}."
        )


def build_aggregation_exprs(columns: List[str], agg_names: List[str]) -> List[pl.Expr]:
    """
    Build the aggregation expressions for the given feature columns.

    :param columns: The feature columns to aggregate.
    :type columns: List[str]
    :param agg_names: The (validated) aggregation names to apply per column.
    :type agg_names: List[str]
    :return: One expression per (column, aggregation) pair.
    :rtype: List[pl.Expr]
    """
    return [
        AGGREGATION_REGISTRY[agg_name](col) for col in columns for agg_name in agg_names
    ]
