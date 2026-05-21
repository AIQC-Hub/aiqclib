"""
Normalization utilities.

This module centralises the logic shared by every feature class and the feature
extraction step when applying normalization. It supports four normalization
"types", each selected per-feature via ``stats_set.type`` in the
``feature_param_sets`` section of a configuration file:

- ``raw``: no normalization (the default).
- ``min_max``: min-max scaling using values supplied **by hand** in the
  ``feature_stats_sets`` section of the config. This is the historical
  behaviour and is kept unchanged.
- ``auto_min_max``: min-max scaling using min/max values **derived
  automatically** from the dataset's summary statistics.
- ``standard``: standard scaling ``(x - mean) / sd`` using mean/sd values
  derived automatically from the dataset's summary statistics.

For ``auto_min_max`` and ``standard`` the derived values are written to a YAML
normalization file during dataset preparation and re-loaded during
classification, so the same fitted normalization is applied at classification
time without re-entering any values (and without access to the original
training data).

The helpers here are deliberately small and pure so they can be unit-tested
with synthetic Polars frames, independently of the wider pipeline.
"""

import os
from typing import Dict, List, Optional

import polars as pl
import yaml

#: Normalization types that actually transform feature values. ``raw`` is
#: intentionally excluded because it is a no-op.
SCALING_TYPES = ("min_max", "auto_min_max", "standard")

#: Normalization types whose values are derived from data (and therefore must
#: be persisted to a normalization file for reuse at classification time),
#: as opposed to ``min_max`` whose values are supplied directly in the config.
AUTO_SCALING_TYPES = ("auto_min_max", "standard")


def is_scaling_type(stats_type: Optional[str]) -> bool:
    """
    Return whether a given ``stats_set`` type performs a value transformation.

    :param stats_type: The normalization type (e.g. ``"min_max"``, ``"raw"``).
    :type stats_type: Optional[str]
    :returns: ``True`` for ``min_max``, ``auto_min_max`` and ``standard``;
              ``False`` for ``raw``, ``None`` and any unknown value.
    :rtype: bool
    """
    return stats_type in SCALING_TYPES


def build_scaling_expr(col_name: str, params: Dict, stats_type: str) -> pl.Expr:
    """
    Build a Polars expression that normalizes a single column.

    The formula depends on ``stats_type``:

    - ``min_max`` / ``auto_min_max``: ``(x - min) / (max - min)``
    - ``standard``: ``(x - mean) / sd``

    A zero denominator (a constant column, e.g. a per-profile location whose
    standard deviation is zero) is handled gracefully by only subtracting the
    centre, which yields ``0`` for the constant value rather than ``inf``/``nan``.

    :param col_name: Name of the column to scale (the output keeps the name).
    :type col_name: str
    :param params: The statistics for this column. For min-max types this is
                   ``{"min": ..., "max": ...}``; for ``standard`` it is
                   ``{"mean": ..., "sd": ...}``.
    :type params: Dict
    :param stats_type: One of :data:`SCALING_TYPES`.
    :type stats_type: str
    :returns: A Polars expression aliased back to ``col_name``.
    :rtype: pl.Expr
    """
    if stats_type in ("min_max", "auto_min_max"):
        lo = params["min"]
        hi = params["max"]
        denom = hi - lo
        if denom == 0:
            return (pl.col(col_name) - lo).alias(col_name)
        return ((pl.col(col_name) - lo) / denom).alias(col_name)

    if stats_type == "standard":
        mean = params["mean"]
        sd = params["sd"]
        if sd == 0:
            return (pl.col(col_name) - mean).alias(col_name)
        return ((pl.col(col_name) - mean) / sd).alias(col_name)

    # Unknown / raw: leave the column untouched.
    return pl.col(col_name)


def scale_flat_columns(
    df: pl.DataFrame, stats: Dict[str, Dict], stats_type: str
) -> pl.DataFrame:
    """
    Apply scaling to a frame whose stats are keyed directly by column name.

    Used by features that operate on raw observed variables (e.g.
    ``basic_values``, ``flank_up``, ``flank_down``, ``location``), where
    ``stats`` looks like ``{"temp": {"min": ..., "max": ...}, ...}``.

    Columns present in ``stats`` but absent from ``df`` are skipped, so a single
    shared stats set can be reused across features that expose different subsets
    of columns.

    :param df: The frame to transform.
    :type df: pl.DataFrame
    :param stats: Mapping of column name to its statistics.
    :type stats: Dict[str, Dict]
    :param stats_type: One of :data:`SCALING_TYPES`.
    :type stats_type: str
    :returns: A new frame with the relevant columns scaled.
    :rtype: pl.DataFrame
    """
    exprs = [
        build_scaling_expr(col_name, params, stats_type)
        for col_name, params in stats.items()
        if col_name in df.columns
    ]
    if not exprs:
        return df
    return df.with_columns(exprs)


def scale_nested_columns(
    df: pl.DataFrame, stats: Dict[str, Dict], stats_type: str
) -> pl.DataFrame:
    """
    Apply scaling to a frame whose stats are keyed by ``variable`` then ``stat``.

    Used by ``profile_summary_stats``, whose feature columns are named
    ``{variable}_{stat}`` (e.g. ``temp_mean``) and whose ``stats`` looks like
    ``{"temp": {"mean": {"min": ..., "max": ...}, ...}, ...}``.

    Columns derived from the nested keys but absent from ``df`` are skipped.

    :param df: The frame to transform.
    :type df: pl.DataFrame
    :param stats: Nested mapping ``{variable: {stat: stats_dict}}``.
    :type stats: Dict[str, Dict]
    :param stats_type: One of :data:`SCALING_TYPES`.
    :type stats_type: str
    :returns: A new frame with the relevant columns scaled.
    :rtype: pl.DataFrame
    """
    exprs = []
    for variable, per_stat in stats.items():
        for stat_name, params in per_stat.items():
            col_name = f"{variable}_{stat_name}"
            if col_name in df.columns:
                exprs.append(build_scaling_expr(col_name, params, stats_type))
    if not exprs:
        return df
    return df.with_columns(exprs)


def aggregate_profile_stats(
    summary_stats: pl.DataFrame,
    variables: Optional[List[str]] = None,
    exclude: List[str] = ["longitude", "latitude"],
) -> pl.DataFrame:
    """
    Aggregate per-profile summary statistics across profiles.

    This reshapes the long per-profile rows of a ``summary_stats`` table (i.e.
    the rows whose ``platform_code`` is not ``"all"``) into one row per
    ``(variable, stats)`` pair, computing the distribution of each per-profile
    statistic **across profiles**: its ``min``, ``mean``, ``pct97.5``, ``max``
    and ``sd``.

    The across-profile ``sd`` is the only addition relative to the historical
    ``SummaryStatsBase.create_summary_stats_profile`` output; it is required to
    standard-scale ``profile_summary_stats`` features (whose columns are
    themselves per-profile statistics).

    :param summary_stats: The combined summary statistics table produced by
                          :meth:`SummaryStatsBase.calculate_stats`.
    :type summary_stats: pl.DataFrame
    :param variables: Optional list of variables to keep. ``None`` keeps all.
    :type variables: Optional[List[str]]
    :param exclude: Variables to drop before aggregating (location variables
                    have no meaningful per-profile spread).
    :type exclude: List[str]
    :returns: A long-form frame with columns ``variable``, ``stats``, ``min``,
              ``mean``, ``pct97.5``, ``max`` and ``sd``.
    :rtype: pl.DataFrame
    """
    df = summary_stats.filter(pl.col("platform_code") != "all")
    if exclude:
        df = df.filter(~pl.col("variable").is_in(exclude))
    if variables is not None:
        df = df.filter(pl.col("variable").is_in(variables))

    return (
        df.unpivot(
            index=["platform_code", "profile_no", "variable"], variable_name="stats"
        )
        .group_by(["variable", "stats"])
        .agg(
            min=pl.col("value").min(),
            mean=pl.col("value").mean(),
            pct97_5=pl.col("value").quantile(0.975),
            max=pl.col("value").max(),
            sd=pl.col("value").std(),
        )
        .rename({"pct97_5": "pct97.5"})
        .sort(["variable", "stats"])
    )


def derive_observation_stats(
    summary_stats: pl.DataFrame, variables: List[str], stats_type: str
) -> Dict[str, Dict]:
    """
    Derive flat per-variable normalization stats from the global ("all") rows.

    For each requested variable this reads its global summary row
    (``platform_code == "all"``) and extracts either ``{min, max}``
    (for ``auto_min_max``) or ``{mean, sd}`` (for ``standard``).

    :param summary_stats: The combined summary statistics table.
    :type summary_stats: pl.DataFrame
    :param variables: The variables (column names) to derive stats for.
    :type variables: List[str]
    :param stats_type: ``"auto_min_max"`` or ``"standard"``.
    :type stats_type: str
    :returns: ``{variable: {"min"/"max"} or {"mean"/"sd"}}``.
    :rtype: Dict[str, Dict]
    """
    all_rows = summary_stats.filter(pl.col("platform_code") == "all")
    out: Dict[str, Dict] = {}
    for variable in variables:
        match = all_rows.filter(pl.col("variable") == variable)
        if match.height == 0:
            continue
        row = match.row(0, named=True)
        out[variable] = _stats_entry(row, stats_type)
    return out


def derive_profile_stats(
    profile_stats_long: pl.DataFrame,
    variables: List[str],
    summary_stats_names: List[str],
    stats_type: str,
) -> Dict[str, Dict]:
    """
    Derive nested per-(variable, stat) normalization stats across profiles.

    Used for ``profile_summary_stats`` features. ``profile_stats_long`` is the
    output of :func:`aggregate_profile_stats`; for each requested variable and
    each requested per-profile statistic, this extracts ``{min, max}`` (for
    ``auto_min_max``) or ``{mean, sd}`` (for ``standard``).

    :param profile_stats_long: The across-profile aggregation.
    :type profile_stats_long: pl.DataFrame
    :param variables: The variables (e.g. ``["temp", "psal", "pres"]``).
    :type variables: List[str]
    :param summary_stats_names: The per-profile statistics that become feature
                                columns (e.g. ``["mean", "median", "sd"]``).
    :type summary_stats_names: List[str]
    :param stats_type: ``"auto_min_max"`` or ``"standard"``.
    :type stats_type: str
    :returns: ``{variable: {stat: {"min"/"max"} or {"mean"/"sd"}}}``.
    :rtype: Dict[str, Dict]
    """
    out: Dict[str, Dict] = {}
    for row in profile_stats_long.iter_rows(named=True):
        if row["variable"] not in variables:
            continue
        if row["stats"] not in summary_stats_names:
            continue
        out.setdefault(row["variable"], {})[row["stats"]] = _stats_entry(
            row, stats_type
        )
    return out


def _stats_entry(row: Dict, stats_type: str) -> Dict:
    """
    Build a single stats entry from a summary row for the given type.

    :param row: A mapping that contains ``min``/``max`` and ``mean``/``sd`` keys.
    :type row: Dict
    :param stats_type: ``"auto_min_max"`` or ``"standard"``.
    :type stats_type: str
    :returns: ``{"min": ..., "max": ...}`` or ``{"mean": ..., "sd": ...}``.
    :rtype: Dict
    :raises ValueError: If ``stats_type`` is not a derivable (auto) type.
    """
    if stats_type == "standard":
        return {"mean": row["mean"], "sd": row["sd"]}
    if stats_type == "auto_min_max":
        return {"min": row["min"], "max": row["max"]}
    raise ValueError(
        f"Cannot derive statistics for non-auto normalization type '{stats_type}'."
    )


def write_normalization_file(
    output_file: str, stats_set_name: str, resolved: Dict[str, Dict[str, Dict]]
) -> None:
    """
    Write derived normalization values to a YAML file.

    The file mirrors the structure of a single ``feature_stats_sets`` entry so
    it can be loaded straight back into a configuration's ``feature_stats_set``
    and consumed by the existing stats-injection machinery. For example::

        name: feature_set_1_stats_set_1
        auto_min_max:
          - name: basic_values3
            stats: {temp: {min: 0.0, max: 20.0}, ...}
        standard:
          - name: location
            stats: {longitude: {mean: 18.8, sd: 2.0}, ...}

    :param output_file: Destination path. Parent directories are created.
    :type output_file: str
    :param stats_set_name: The ``name`` recorded at the top of the file.
    :type stats_set_name: str
    :param resolved: ``{stats_type: {entry_name: stats_dict}}`` to serialise.
    :type resolved: Dict[str, Dict[str, Dict]]
    :returns: None
    :rtype: None
    """
    document: Dict = {"name": stats_set_name}
    for stats_type, entries in resolved.items():
        document[stats_type] = [
            {"name": name, "stats": stats} for name, stats in entries.items()
        ]

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as handle:
        yaml.safe_dump(document, handle, sort_keys=False, default_flow_style=None)


def read_normalization_file(input_file: str) -> Dict:
    """
    Read a normalization YAML file written by :func:`write_normalization_file`.

    :param input_file: Path to the YAML normalization file.
    :type input_file: str
    :raises FileNotFoundError: If the file does not exist.
    :returns: A dictionary shaped like a ``feature_stats_set`` entry (i.e. with
              ``name`` plus ``auto_min_max`` / ``standard`` lists).
    :rtype: Dict
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(
            f"Normalization file '{input_file}' does not exist. It is produced "
            f"during dataset preparation when 'auto_min_max' or 'standard' "
            f"normalization is used."
        )
    with open(input_file, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)
