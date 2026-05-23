"""
Tests for :mod:`aiqclib.common.utils.input_validation` and the input step's
:meth:`InputDataSetBase.validate_input_columns` hook.

All data is synthetic, so these run without the downloaded ``tests/data``
fixtures.
"""

from datetime import date, datetime

import polars as pl
import pytest

from aiqclib.common.config.dataset_config import DataSetConfig
from aiqclib.common.utils.input_validation import (
    REQUIRED_INPUT_COLUMNS,
    required_column_names,
    validate_and_convert_input_columns,
)
from aiqclib.prepare.step1_read_input.dataset_a import InputDataSetA


def _good_frame():
    """A frame already in the expected types."""
    return pl.DataFrame(
        {
            "platform_code": ["A", "B"],
            "profile_no": [1, 2],
            "profile_timestamp": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
            "longitude": [10.0, -1.5],
            "latitude": [45.0, 40.0],
            "observation_no": [1, 2],
            "pres": [1.0, 2.0],
            "temp": [15.1, 20.0],  # extra columns are left untouched
        }
    )


def test_registry_matches_documented_columns():
    assert required_column_names() == [
        "platform_code",
        "profile_no",
        "profile_timestamp",
        "longitude",
        "latitude",
        "observation_no",
        "pres",
    ]


def test_already_valid_frame_is_unchanged():
    df = _good_frame()
    out = validate_and_convert_input_columns(df)
    assert out.schema == df.schema
    assert out["temp"].to_list() == [15.1, 20.0]  # extras preserved


def test_auto_conversion_from_string_like_inputs():
    # Mimics a CSV/TSV read where everything came in as strings/ints.
    df = pl.DataFrame(
        {
            "platform_code": [1, 2],  # int -> text
            "profile_no": ["10", "11"],  # str -> integer
            "profile_timestamp": [
                "2020-01-02 03:04:05",
                "2021-06-07 08:09:10",
            ],  # str -> datetime
            "longitude": ["10.5", "-1.5"],  # str -> float
            "latitude": [45, 40],  # int -> float
            "observation_no": ["1", "2"],  # str -> int
            "pres": [1, 2],  # int -> float
        }
    )
    out = validate_and_convert_input_columns(df)

    assert out.schema["platform_code"] == pl.Utf8
    assert out.schema["profile_no"] == pl.Int64
    assert isinstance(out.schema["profile_timestamp"], pl.Datetime)
    assert out.schema["longitude"] == pl.Float64
    assert out.schema["latitude"] == pl.Float64
    assert out.schema["observation_no"] == pl.Int64
    assert out.schema["pres"] == pl.Float64

    # Values survive conversion.
    assert out["platform_code"].to_list() == ["1", "2"]
    assert out["longitude"].to_list() == [10.5, -1.5]
    assert out["observation_no"].to_list() == [1, 2]
    assert out["profile_timestamp"].to_list()[0] == datetime(2020, 1, 2, 3, 4, 5)


def test_date_is_promoted_to_datetime():
    df = _good_frame().with_columns(
        pl.Series("profile_timestamp", [date(2020, 1, 1), date(2020, 1, 2)], dtype=pl.Date)
    )
    out = validate_and_convert_input_columns(df)
    assert isinstance(out.schema["profile_timestamp"], pl.Datetime)


def test_missing_columns_raise_with_names():
    df = _good_frame().drop(["latitude", "pres"])
    with pytest.raises(ValueError) as excinfo:
        validate_and_convert_input_columns(df)
    msg = str(excinfo.value)
    assert "latitude" in msg and "pres" in msg


def test_numeric_timestamp_raises_helpful_error():
    # A float "days since epoch" encoding is ambiguous and must be rejected.
    df = _good_frame().with_columns(
        pl.Series("profile_timestamp", [14688.58, 14690.0], dtype=pl.Float64)
    )
    with pytest.raises(ValueError) as excinfo:
        validate_and_convert_input_columns(df)
    assert "profile_timestamp" in str(excinfo.value)


def test_unconvertible_value_raises_with_column_name():
    df = _good_frame().with_columns(
        pl.Series("longitude", ["not_a_number", "x"], dtype=pl.Utf8)
    )
    with pytest.raises(ValueError) as excinfo:
        validate_and_convert_input_columns(df)
    assert "longitude" in str(excinfo.value)


def test_custom_required_columns_table():
    df = pl.DataFrame({"only_col": [1, 2]})
    out = validate_and_convert_input_columns(df, required_columns={"only_col": "float"})
    assert out.schema["only_col"] == pl.Float64


# --------------------------------------------------------------------------- #
# Integration with the input step (covers both prepare and classify, which
# share this base method).
# --------------------------------------------------------------------------- #
def _config():
    config = DataSetConfig("template:data_sets")
    config.select("dataset_0001")
    return config


def test_input_step_validates_after_rename():
    extractor = InputDataSetA(_config())
    extractor.input_data = pl.DataFrame(
        {
            "platform_code": ["A"],
            "profile_no": [1],
            "profile_timestamp": ["2020-01-01 00:00:00"],
            "longitude": [10],
            "latitude": [45],
            "observation_no": ["1"],
            "pres": [1],
        }
    )
    extractor.validate_input_columns()
    assert isinstance(extractor.input_data.schema["profile_timestamp"], pl.Datetime)
    assert extractor.input_data.schema["pres"] == pl.Float64
    assert extractor.input_data.schema["observation_no"] == pl.Int64


def test_input_step_validation_can_be_disabled():
    config = _config()
    config.get_step_params("input").setdefault("sub_steps", {})[
        "validate_columns"
    ] = False

    extractor = InputDataSetA(config)
    # 'pres' deliberately stays an int; with validation off it must be untouched.
    extractor.input_data = pl.DataFrame(
        {
            "platform_code": ["A"],
            "profile_no": [1],
            "profile_timestamp": [datetime(2020, 1, 1)],
            "longitude": [10.0],
            "latitude": [45.0],
            "observation_no": [1],
            "pres": [1],
        }
    )
    extractor.validate_input_columns()
    assert extractor.input_data.schema["pres"] == pl.Int64  # unchanged
