"""
Tests for :mod:`aiqclib.common.utils.input_preprocess` and the input step's
:meth:`InputDataSetBase.create_columns` hook.

All data is synthetic, so these run without the downloaded ``tests/data``
fixtures.
"""

from datetime import datetime

import polars as pl
import pytest

from aiqclib.common.config.dataset_config import DataSetConfig
from aiqclib.common.utils.input_preprocess import create_identifier_columns
from aiqclib.prepare.step1_read_input.dataset_a import InputDataSetA


def _raw_frame():
    """Two platforms; A has two profiles, B has one. Rows intentionally
    unsorted within profiles so ordering can be verified."""
    return pl.DataFrame(
        {
            "platform_code": ["A", "A", "A", "B", "B"],
            "profile_timestamp": [
                datetime(2020, 1, 2),  # A profile 2
                datetime(2020, 1, 1),  # A profile 1
                datetime(2020, 1, 1),  # A profile 1
                datetime(2020, 1, 1),  # B profile 1
                datetime(2020, 1, 1),  # B profile 1
            ],
            "longitude": [11.0, 10.0, 10.0, 5.0, 5.0],
            "latitude": [46.0, 45.0, 45.0, 40.0, 40.0],
            "pres": [1.0, 2.0, 1.0, 2.0, 1.0],
        }
    )


def test_creates_both_identifier_columns():
    out = create_identifier_columns(_raw_frame())

    assert out.schema["profile_no"] == pl.Int64
    assert out.schema["observation_no"] == pl.Int64
    assert "__profile_key__" not in out.columns  # temp key dropped

    # Three distinct profiles overall: (A,1), (A,2), (B,1).
    profiles = out.select(["platform_code", "profile_no"]).unique()
    assert profiles.height == 3
    # profile_no restarts per platform.
    assert sorted(
        out.filter(pl.col("platform_code") == "A")["profile_no"].unique().to_list()
    ) == [1, 2]
    assert out.filter(pl.col("platform_code") == "B")[
        "profile_no"
    ].unique().to_list() == [1]


def test_observation_no_is_one_indexed_and_pressure_ordered():
    out = create_identifier_columns(_raw_frame())
    # Profile A/1 had pres 2.0 and 1.0; after numbering, obs 1 -> pres 1.0.
    a1 = out.filter(
        (pl.col("platform_code") == "A") & (pl.col("profile_no") == 1)
    ).sort("observation_no")
    assert a1["observation_no"].to_list() == [1, 2]
    assert a1["pres"].to_list() == [1.0, 2.0]


def test_string_pressure_is_sorted_numerically():
    df = pl.DataFrame(
        {
            "platform_code": ["A", "A", "A"],
            "profile_timestamp": [datetime(2020, 1, 1)] * 3,
            "longitude": [10.0, 10.0, 10.0],
            "latitude": [45.0, 45.0, 45.0],
            "pres": ["2", "10", "1"],  # string -> must sort as 1,2,10 not 1,10,2
        }
    )
    out = create_identifier_columns(df).sort("observation_no")
    assert out["pres"].to_list() == [1.0, 2.0, 10.0]
    assert out["observation_no"].to_list() == [1, 2, 3]


def test_create_only_observation_no():
    out = create_identifier_columns(_raw_frame(), columns=["observation_no"])
    assert "observation_no" in out.columns
    assert "profile_no" not in out.columns


def test_create_only_profile_no():
    out = create_identifier_columns(_raw_frame(), columns=["profile_no"])
    assert "profile_no" in out.columns
    assert "observation_no" not in out.columns


def test_missing_source_column_raises():
    df = _raw_frame().drop("latitude")
    with pytest.raises(ValueError) as excinfo:
        create_identifier_columns(df)
    assert "latitude" in str(excinfo.value)


def test_custom_key_and_sort_columns():
    df = pl.DataFrame(
        {
            "platform_code": ["A", "A"],
            "profile_timestamp": [datetime(2020, 1, 1)] * 2,
            "longitude": [10.0, 10.0],
            "latitude": [45.0, 45.0],
            "pres": [1.0, 2.0],
            "cast_no": [7, 7],
        }
    )
    # Use cast_no as the only profile key; both rows share it -> one profile.
    out = create_identifier_columns(
        df,
        key_columns=["platform_code", "cast_no"],
        sort_columns=["platform_code", "cast_no", "pres"],
    )
    assert out["profile_no"].unique().to_list() == [1]
    assert out.sort("observation_no")["observation_no"].to_list() == [1, 2]


# --------------------------------------------------------------------------- #
# Integration with the input step (shared by prepare and classify).
# --------------------------------------------------------------------------- #
def _config(create_columns=True, create_column_dict=None):
    config = DataSetConfig("template:data_sets")
    config.select("dataset_0001")
    sub_steps = config.get_step_params("input").setdefault("sub_steps", {})
    sub_steps["create_columns"] = create_columns
    if create_column_dict is not None:
        config.get_step_params("input")["create_column_dict"] = create_column_dict
    return config


def test_input_step_creates_then_validates():
    extractor = InputDataSetA(_config(create_columns=True))
    # No profile_no / observation_no present in the raw input.
    extractor.input_data = _raw_frame()
    extractor.create_columns()
    assert "profile_no" in extractor.input_data.columns
    assert "observation_no" in extractor.input_data.columns

    # The created columns survive validation as integers.
    extractor.validate_input_columns()
    assert extractor.input_data.schema["profile_no"] == pl.Int64
    assert extractor.input_data.schema["observation_no"] == pl.Int64


def test_input_step_creation_disabled_by_default():
    extractor = InputDataSetA(_config(create_columns=False))
    extractor.input_data = _raw_frame()
    extractor.create_columns()
    assert "profile_no" not in extractor.input_data.columns
    assert "observation_no" not in extractor.input_data.columns


def test_input_step_create_column_dict_is_honored():
    extractor = InputDataSetA(
        _config(create_column_dict={"columns": ["observation_no"]})
    )
    extractor.input_data = _raw_frame()
    extractor.create_columns()
    assert "observation_no" in extractor.input_data.columns
    assert "profile_no" not in extractor.input_data.columns
