"""
Integration tests for the data-derived normalization flow.

These exercise the real configuration objects and the real feature-extraction
step (``ExtractDataSetA``) in both roles:

- ``fit`` (preparation): derive ``auto_min_max`` / ``standard`` values from the
  summary statistics, inject them into the feature parameters and write them to
  the normalization file.
- ``apply`` (classification): read those values back from the file and inject
  them, *without* consulting any summary statistics.

They also confirm that the extended per-profile summary aggregation now exposes
the across-profile ``sd`` needed to standard-scale ``profile_summary_stats``.

All data is synthetic, so these run without the downloaded ``tests/data``
fixtures.
"""

import polars as pl
import pytest

from aiqclib.common.config.dataset_config import DataSetConfig
from aiqclib.common.utils.normalization import read_normalization_file
from aiqclib.prepare.step5_extract_features.dataset_a import ExtractDataSetA

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
    record = {
        "platform_code": platform_code,
        "profile_no": profile_no,
        "variable": variable,
    }
    for col in _STAT_COLS:
        record[col] = float(stats.get(col, 0.0))
    return record


def _synthetic_summary_stats():
    """Three profiles for temp/psal/pres plus global ('all') population rows."""
    rows = []
    # Per-profile rows: temp_mean spans 10/20/30 -> across-profile mean 20, sd 10.
    for prof, base in enumerate([10.0, 20.0, 30.0], start=1):
        for var, scale in [("temp", 1.0), ("psal", 0.1), ("pres", 10.0)]:
            rows.append(
                _row(
                    "A",
                    prof,
                    var,
                    mean=base * scale,
                    median=base * scale,
                    sd=prof * scale,
                    pct25=(base - 1) * scale,
                    pct75=(base + 1) * scale,
                    min=(base - 5) * scale,
                    max=(base + 5) * scale,
                )
            )
    # Global population rows used for observation-level (basic_values) stats.
    rows += [
        _row("all", 0, "temp", min=0, max=40, mean=20, sd=8),
        _row("all", 0, "psal", min=3, max=4, mean=3.5, sd=0.5),
        _row("all", 0, "pres", min=0, max=1000, mean=500, sd=100),
        _row("all", 0, "longitude", min=10, max=25, mean=18, sd=2),
        _row("all", 0, "latitude", min=30, max=50, mean=40, sd=5),
    ]
    return pl.DataFrame(rows)


def _make_config(tmp_path):
    """Build a template dataset config rooted at ``tmp_path`` with two
    data-derived features: basic_values -> standard, profile_summary_stats ->
    auto_min_max."""
    config = DataSetConfig("template:data_sets")
    config.select("dataset_0001")
    config.data["path_info"]["common"]["base_path"] = str(tmp_path)

    for param in config.data["feature_param_set"]["params"]:
        if param.get("feature") == "basic_values":
            param["stats_set"] = {"type": "standard", "name": "basic_values_stats"}
        elif param.get("feature") == "profile_summary_stats":
            param["stats_set"] = {"type": "auto_min_max", "name": "psum_stats"}
    return config


def _param(config, feature_name):
    for param in config.data["feature_param_set"]["params"]:
        if param.get("feature") == feature_name:
            return param
    raise AssertionError(f"feature {feature_name} not found")


def test_fit_writes_file_and_injects_stats(tmp_path):
    config = _make_config(tmp_path)
    extractor = ExtractDataSetA(config, summary_stats=_synthetic_summary_stats())
    assert extractor.normalization_role == "fit"

    extractor.apply_normalization()

    # 1) basic_values (standard, observation-level) got flat mean/sd from 'all'.
    basic_stats = _param(config, "basic_values")["stats"]
    assert basic_stats["temp"] == {"mean": 20.0, "sd": 8.0}
    assert basic_stats["pres"] == {"mean": 500.0, "sd": 100.0}

    # 2) profile_summary_stats (auto_min_max, nested) got per-(var, stat) min/max
    #    derived across profiles. temp_mean spanned 10..30.
    psum_stats = _param(config, "profile_summary_stats")["stats"]
    assert psum_stats["temp"]["mean"] == {"min": 10.0, "max": 30.0}
    # only the configured summary_stats_names appear
    assert set(psum_stats["temp"].keys()) == {"mean", "median", "sd", "pct25", "pct75"}

    # 3) the normalization file was written with both sections.
    norm_file = config.get_normalization_file_name()
    loaded = read_normalization_file(norm_file)
    assert {e["name"] for e in loaded["standard"]} == {"basic_values_stats"}
    assert {e["name"] for e in loaded["auto_min_max"]} == {"psum_stats"}


def test_apply_reads_file_without_summary_stats(tmp_path):
    # First, fit to produce the file.
    fit_config = _make_config(tmp_path)
    ExtractDataSetA(
        fit_config, summary_stats=_synthetic_summary_stats()
    ).apply_normalization()
    norm_file = fit_config.get_normalization_file_name()

    # Now a fresh config + an apply-mode extractor with NO summary stats.
    apply_config = _make_config(tmp_path)
    assert apply_config.get_normalization_file_name() == norm_file

    extractor = ExtractDataSetA(apply_config, summary_stats=None)
    extractor.normalization_role = "apply"
    extractor.apply_normalization()  # must succeed using only the file

    basic_stats = _param(apply_config, "basic_values")["stats"]
    assert basic_stats["temp"] == {"mean": 20.0, "sd": 8.0}

    psum_stats = _param(apply_config, "profile_summary_stats")["stats"]
    assert psum_stats["temp"]["mean"] == {"min": 10.0, "max": 30.0}


def test_no_auto_features_writes_no_file(tmp_path):
    """A config using only raw/min_max must not write a normalization file."""
    config = DataSetConfig("template:data_sets")
    config.select("dataset_0001")
    config.data["path_info"]["common"]["base_path"] = str(tmp_path)
    # template:data_sets uses only 'raw' features by default.

    extractor = ExtractDataSetA(config, summary_stats=_synthetic_summary_stats())
    extractor.apply_normalization()

    import os

    assert not os.path.exists(config.get_normalization_file_name())


def test_summary_step_profile_aggregation_exposes_sd(tmp_path):
    """The extended per-profile aggregation surfaces the across-profile sd."""
    from aiqclib.common.utils.normalization import aggregate_profile_stats

    agg = aggregate_profile_stats(_synthetic_summary_stats())
    assert "sd" in agg.columns
    # temp per-profile 'mean' statistic: values 10/20/30 across profiles.
    row = agg.filter((pl.col("variable") == "temp") & (pl.col("stats") == "mean")).row(
        0, named=True
    )
    assert row["mean"] == pytest.approx(20.0)
    assert row["sd"] == pytest.approx(10.0)
