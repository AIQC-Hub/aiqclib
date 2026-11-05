"""Regression tests: every NRT QC item treats NaN exactly like null.

Inputs such as ctddump write missing measurements as floating-point NaN.
Polars orders NaN above every number, so before the fix a NaN failed the
range tests and poisoned the neighbour comparisons of the stencil tests.
``QCItemFeatureBase.extract_features`` now converts NaN to null first; these
tests check that each item gives the same flags for NaN as for null.
"""

from datetime import datetime

import polars as pl
import pytest

from aiqclib.common.loader.feature_registry import FEATURE_REGISTRY
from aiqclib.prepare.features.qc_item_base import QCItemFeatureBase

NAN = float("nan")
N_OBS = 7

#: (item, feature_info) for every QC item with float inputs.
ITEMS = [
    ("qc_impossible_location", None),
    ("qc_position_on_land", None),
    ("qc_position_on_land", {"params": {"positive_depth": False}}),
    ("qc_global_range", None),
    (
        "qc_regional_range",
        {"params": {"temp": {"min": -2.0, "max": 24.0}, "psal": {"min": 0, "max": 38}}},
    ),
    ("qc_pressure_increasing", None),
    ("qc_spike", None),
    ("qc_gradient", None),
    ("qc_digit_rollover", None),
    ("qc_stuck_value", None),
    ("qc_density_inversion", None),
]

#: The float columns a missing value is injected into.
COLUMNS = ["temp", "psal", "pres", "latitude", "longitude", "bathymetry"]


def make_profile(missing=None, column: str = None, rows=(3,)) -> pl.DataFrame:
    """Build one profile, with ``missing`` written into ``column`` at ``rows``."""
    data = {
        "temp": [10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6],
        "psal": [35.0, 35.01, 35.02, 35.03, 35.04, 35.05, 35.06],
        "pres": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        "latitude": [55.0] * N_OBS,
        "longitude": [15.0] * N_OBS,
        "bathymetry": [100.0] * N_OBS,
    }
    if column is not None:
        data[column] = [missing if i in rows else v for i, v in enumerate(data[column])]
    return pl.DataFrame(
        {
            "platform_code": ["P1"] * N_OBS,
            "profile_no": [1] * N_OBS,
            "observation_no": list(range(1, N_OBS + 1)),
            "profile_timestamp": pl.Series(
                [datetime(2023, 6, 1, 12)] * N_OBS, dtype=pl.Datetime("ms")
            ),
            **{k: pl.Series(v, dtype=pl.Float64) for k, v in data.items()},
        }
    )


def run_item(name: str, df: pl.DataFrame, feature_info: dict = None) -> pl.DataFrame:
    """Run a registered QC item in full-frame mode and return its features."""
    ds = FEATURE_REGISTRY[name](feature_info=feature_info, filtered_input=df)
    ds.extract_features()
    return ds.features


@pytest.mark.parametrize("column", COLUMNS)
@pytest.mark.parametrize("name, feature_info", ITEMS)
def test_nan_flags_equal_null_flags(name, feature_info, column):
    """A NaN gives the same flags as a null in every item and column."""
    nan_flags = run_item(name, make_profile(NAN, column), feature_info)
    null_flags = run_item(name, make_profile(None, column), feature_info)
    assert nan_flags.equals(null_flags)


@pytest.mark.parametrize("name, feature_info", ITEMS)
def test_whole_variable_nan_equals_null(name, feature_info):
    """A variable that is NaN throughout behaves like an unmeasured one."""
    rows = range(N_OBS)
    nan_flags = run_item(name, make_profile(NAN, "temp", rows), feature_info)
    null_flags = run_item(name, make_profile(None, "temp", rows), feature_info)
    assert nan_flags.equals(null_flags)


@pytest.mark.parametrize(
    "name", ["qc_global_range", "qc_spike", "qc_gradient", "qc_digit_rollover"]
)
def test_nan_measurement_and_neighbours_pass(name):
    """The NaN value and its neighbours pass instead of failing."""
    flags = run_item(name, make_profile(NAN, "temp"))
    assert flags[f"temp_{name}"].to_list() == [1] * N_OBS


def test_all_nan_variable_is_not_stuck():
    flags = run_item("qc_stuck_value", make_profile(NAN, "temp", range(N_OBS)))
    assert flags["temp_qc_stuck_value"].to_list() == [1] * N_OBS


@pytest.mark.parametrize("positive_depth", [True, False])
def test_nan_depth_passes_either_convention(positive_depth):
    flags = run_item(
        "qc_position_on_land",
        make_profile(NAN, "bathymetry", range(N_OBS)),
        {"params": {"positive_depth": positive_depth}},
    )
    assert flags["qc_position_on_land"].to_list() == [1] * N_OBS


@pytest.mark.parametrize("column", ["latitude", "longitude"])
def test_nan_location_still_fails(column):
    """A missing position fails, whether it is NaN or null."""
    flags = run_item("qc_impossible_location", make_profile(NAN, column, range(N_OBS)))
    assert flags["qc_impossible_location"].to_list() == [4] * N_OBS


def test_selected_rows_mode_treats_nan_as_null():
    """The preparation pipeline path picks up the fix too."""
    df = make_profile(NAN, "temp").with_row_index("row_id")
    selected = {
        "temp": df.select(["row_id", "platform_code", "profile_no", "observation_no"])
    }
    ds = FEATURE_REGISTRY["qc_spike"](
        target_name="temp", filtered_input=df.drop("row_id"), selected_rows=selected
    )
    ds.extract_features()
    assert ds.features["temp_qc_spike"].to_list() == [1] * N_OBS


def test_missing_as_null_only_touches_floats():
    df = pl.DataFrame({"x": [1.0, NAN, None], "n": [1, 2, 3], "s": ["a", "b", "c"]})
    out = QCItemFeatureBase.missing_as_null(df)
    assert out["x"].to_list() == [1.0, None, None]
    assert out["n"].to_list() == [1, 2, 3]
    assert out["s"].to_list() == ["a", "b", "c"]
    assert out.schema == df.schema
