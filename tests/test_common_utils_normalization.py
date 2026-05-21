"""
Unit tests for :mod:`aiqclib.common.utils.normalization`.

These tests are deliberately self-contained: they build small synthetic Polars
frames in-memory and assert exact numeric results, so they run without any of
the downloaded ``tests/data`` fixtures.
"""

import math

import polars as pl
import pytest

from aiqclib.common.utils.normalization import (
    AUTO_SCALING_TYPES,
    SCALING_TYPES,
    aggregate_profile_stats,
    build_scaling_expr,
    derive_observation_stats,
    derive_profile_stats,
    is_scaling_type,
    read_normalization_file,
    scale_flat_columns,
    scale_nested_columns,
    write_normalization_file,
)


# Synthetic summary-statistics columns mirroring the real ``summary_stats`` table.
_STAT_COLS = [
    "min",
    "pct2.5",
    "pct25",
    "mean",
    "median",
    "pct75",
    "pct97.5",
    "max",
    "sd",
]


def _row(platform_code, profile_no, variable, **stats):
    """Build one summary-stats row, defaulting unspecified stat columns to 0.0."""
    record = {
        "platform_code": platform_code,
        "profile_no": profile_no,
        "variable": variable,
    }
    for col in _STAT_COLS:
        record[col] = float(stats.get(col, 0.0))
    return record


@pytest.fixture
def summary_stats():
    """
    A deterministic synthetic ``summary_stats`` frame.

    ``temp`` has three profiles whose per-profile ``mean`` values are 10/20/30
    (so the across-profile mean is 20 and the sample sd is 10) and whose
    per-profile ``sd`` values are 1/2/3. The global ("all") rows carry the
    population statistics used for observation-level normalization.
    """
    rows = [
        _row("A", 1, "temp", mean=10, median=10, sd=1, pct25=9, pct75=11, min=5, max=15),
        _row("A", 2, "temp", mean=20, median=20, sd=2, pct25=18, pct75=22, min=10, max=30),
        _row("A", 3, "temp", mean=30, median=30, sd=3, pct25=27, pct75=33, min=20, max=40),
        # Location vars carry zero per-profile spread; they must be excluded
        # from across-profile aggregation.
        _row("A", 1, "longitude", mean=18, sd=0),
        _row("A", 2, "longitude", mean=19, sd=0),
        # Global population rows.
        _row("all", 0, "temp", min=0, max=40, mean=20, sd=8),
        _row("all", 0, "psal", min=30, max=40, mean=35, sd=5),
        _row("all", 0, "pres", min=0, max=1000, mean=500, sd=100),
        _row("all", 0, "longitude", min=10, max=25, mean=18, sd=2),
        _row("all", 0, "latitude", min=30, max=50, mean=40, sd=5),
    ]
    return pl.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Type predicate
# --------------------------------------------------------------------------- #
def test_is_scaling_type():
    assert SCALING_TYPES == ("min_max", "auto_min_max", "standard")
    assert AUTO_SCALING_TYPES == ("auto_min_max", "standard")
    for t in SCALING_TYPES:
        assert is_scaling_type(t)
    for t in ("raw", None, "unknown", ""):
        assert not is_scaling_type(t)


# --------------------------------------------------------------------------- #
# Scaling expressions / column helpers
# --------------------------------------------------------------------------- #
def test_build_scaling_expr_min_max_and_standard():
    df = pl.DataFrame({"x": [0.0, 5.0, 10.0]})

    mm = df.select(build_scaling_expr("x", {"min": 0.0, "max": 10.0}, "min_max"))
    assert mm["x"].to_list() == [0.0, 0.5, 1.0]

    # auto_min_max uses the identical formula.
    amm = df.select(build_scaling_expr("x", {"min": 0.0, "max": 10.0}, "auto_min_max"))
    assert amm["x"].to_list() == [0.0, 0.5, 1.0]

    std = df.select(build_scaling_expr("x", {"mean": 5.0, "sd": 5.0}, "standard"))
    assert std["x"].to_list() == [-1.0, 0.0, 1.0]


def test_build_scaling_expr_zero_denominator_is_safe():
    df = pl.DataFrame({"x": [7.0, 7.0]})
    # Constant column under min_max (max == min): centre-only, yields 0.
    mm = df.select(build_scaling_expr("x", {"min": 7.0, "max": 7.0}, "min_max"))
    assert mm["x"].to_list() == [0.0, 0.0]
    # Constant column under standard (sd == 0): centre-only, yields 0.
    std = df.select(build_scaling_expr("x", {"mean": 7.0, "sd": 0.0}, "standard"))
    assert std["x"].to_list() == [0.0, 0.0]


def test_build_scaling_expr_raw_is_noop():
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
    out = df.select(build_scaling_expr("x", {}, "raw"))
    assert out["x"].to_list() == [1.0, 2.0, 3.0]


def test_scale_flat_columns_skips_absent_columns():
    df = pl.DataFrame({"temp": [0.0, 10.0], "psal": [30.0, 40.0]})
    stats = {
        "temp": {"min": 0.0, "max": 10.0},
        "psal": {"min": 30.0, "max": 40.0},
        "pres": {"min": 0.0, "max": 1.0},  # not in df -> skipped, no error
    }
    out = scale_flat_columns(df, stats, "min_max")
    assert out["temp"].to_list() == [0.0, 1.0]
    assert out["psal"].to_list() == [0.0, 1.0]
    assert out.columns == ["temp", "psal"]


def test_scale_nested_columns():
    df = pl.DataFrame({"temp_mean": [10.0, 30.0], "temp_sd": [0.0, 4.0]})
    stats = {
        "temp": {
            "mean": {"mean": 20.0, "sd": 10.0},
            "sd": {"mean": 2.0, "sd": 2.0},
        }
    }
    out = scale_nested_columns(df, stats, "standard")
    assert out["temp_mean"].to_list() == [-1.0, 1.0]
    assert out["temp_sd"].to_list() == [-1.0, 1.0]


# --------------------------------------------------------------------------- #
# Aggregation across profiles (the new ``sd`` is the key addition)
# --------------------------------------------------------------------------- #
def test_aggregate_profile_stats_includes_sd_and_excludes_location_and_all(
    summary_stats,
):
    agg = aggregate_profile_stats(summary_stats)

    # location variables and 'all' rows are excluded
    assert set(agg["variable"].unique().to_list()) == {"temp"}

    # the new across-profile 'sd' column is present
    assert "sd" in agg.columns

    # across-profile distribution of the per-profile 'mean' statistic
    mean_row = agg.filter(
        (pl.col("variable") == "temp") & (pl.col("stats") == "mean")
    ).row(0, named=True)
    assert mean_row["mean"] == pytest.approx(20.0)
    assert mean_row["sd"] == pytest.approx(10.0)  # std([10,20,30], ddof=1)
    assert mean_row["min"] == pytest.approx(10.0)
    assert mean_row["max"] == pytest.approx(30.0)

    # across-profile distribution of the per-profile 'sd' statistic
    sd_row = agg.filter(
        (pl.col("variable") == "temp") & (pl.col("stats") == "sd")
    ).row(0, named=True)
    assert sd_row["mean"] == pytest.approx(2.0)  # mean([1,2,3])
    assert sd_row["sd"] == pytest.approx(1.0)  # std([1,2,3], ddof=1)


# --------------------------------------------------------------------------- #
# Derivation: observation-level (flat) and profile-level (nested)
# --------------------------------------------------------------------------- #
def test_derive_observation_stats_standard_and_auto_min_max(summary_stats):
    std = derive_observation_stats(summary_stats, ["temp", "psal", "pres"], "standard")
    assert std["temp"] == {"mean": 20.0, "sd": 8.0}
    assert std["psal"] == {"mean": 35.0, "sd": 5.0}
    assert std["pres"] == {"mean": 500.0, "sd": 100.0}

    amm = derive_observation_stats(
        summary_stats, ["longitude", "latitude"], "auto_min_max"
    )
    assert amm["longitude"] == {"min": 10.0, "max": 25.0}
    assert amm["latitude"] == {"min": 30.0, "max": 50.0}


def test_derive_observation_stats_ignores_unknown_variable(summary_stats):
    out = derive_observation_stats(summary_stats, ["does_not_exist"], "standard")
    assert out == {}


def test_derive_profile_stats_nested(summary_stats):
    agg = aggregate_profile_stats(summary_stats)
    nested = derive_profile_stats(
        agg, ["temp"], ["mean", "sd"], "standard"
    )
    assert nested["temp"]["mean"] == {"mean": 20.0, "sd": 10.0}
    assert nested["temp"]["sd"] == {"mean": 2.0, "sd": 1.0}
    # statistics not requested are excluded
    assert set(nested["temp"].keys()) == {"mean", "sd"}


# --------------------------------------------------------------------------- #
# YAML persistence round-trip
# --------------------------------------------------------------------------- #
def test_write_then_read_normalization_file_round_trip(tmp_path):
    out_file = tmp_path / "nested" / "normalization_stats.yaml"
    resolved = {
        "standard": {
            "basic_values_stats": {"temp": {"mean": 20.0, "sd": 8.0}},
        },
        "auto_min_max": {
            "location_stats": {"longitude": {"min": 10.0, "max": 25.0}},
        },
    }

    write_normalization_file(str(out_file), "feature_set_1_stats_set_1", resolved)
    assert out_file.exists()  # parent dirs created automatically

    loaded = read_normalization_file(str(out_file))
    assert loaded["name"] == "feature_set_1_stats_set_1"

    standard_entry = loaded["standard"][0]
    assert standard_entry["name"] == "basic_values_stats"
    assert standard_entry["stats"]["temp"] == {"mean": 20.0, "sd": 8.0}

    auto_entry = loaded["auto_min_max"][0]
    assert auto_entry["name"] == "location_stats"
    assert auto_entry["stats"]["longitude"] == {"min": 10.0, "max": 25.0}


def test_read_normalization_file_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_normalization_file(str(tmp_path / "nope.yaml"))
