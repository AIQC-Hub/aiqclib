"""Unit tests for the basic NRT QC item feature classes.

Covers the Phase 3 items (impossible date, impossible location, global
range, regional range, stuck value) and the shared ``QCItemFeatureBase``
plumbing: parameter resolution, ``fail_flag`` handling, full-frame vs
selected-rows output, and registry entries. All tests use small synthetic
profiles so expectations can be verified row by row.
"""

from datetime import datetime

import polars as pl
import pytest

from aiqclib.common.loader.feature_registry import FEATURE_REGISTRY
from aiqclib.prepare.features.qc_global_range import QCGlobalRange
from aiqclib.prepare.features.qc_impossible_date import QCImpossibleDate
from aiqclib.prepare.features.qc_impossible_location import QCImpossibleLocation
from aiqclib.prepare.features.qc_regional_range import QCRegionalRange
from aiqclib.prepare.features.qc_stuck_value import QCStuckValue

VALID_TS = datetime(2023, 6, 1, 12, 0)


def make_profile(
    temp: list,
    psal: list = None,
    platform_code: str = "P1",
    profile_no: int = 1,
    timestamp: datetime = VALID_TS,
    longitude: float = 15.0,
    latitude: float = 55.0,
) -> pl.DataFrame:
    """Build one synthetic profile with the standard input columns."""
    n = len(temp)
    psal = psal if psal is not None else [35.0] * n
    return pl.DataFrame(
        {
            "platform_code": [platform_code] * n,
            "profile_no": [profile_no] * n,
            "observation_no": list(range(1, n + 1)),
            # Explicit dtype: an all-None list must still be a Datetime
            # column, as guaranteed by input validation in the pipeline.
            "profile_timestamp": pl.Series([timestamp] * n, dtype=pl.Datetime("ms")),
            "longitude": [longitude] * n,
            "latitude": [latitude] * n,
            "pres": [float(10 * i) for i in range(1, n + 1)],
            "temp": temp,
            "psal": psal,
        }
    )


def run_item(cls, df: pl.DataFrame, feature_info: dict = None) -> pl.DataFrame:
    """Instantiate a QC item in full-frame mode and return its features."""
    ds = cls(feature_info=feature_info, filtered_input=df)
    ds.extract_features()
    return ds.features


# ---------------------------------------------------------------------------
# RTQC2: impossible date
# ---------------------------------------------------------------------------


class TestImpossibleDate:
    """RTQC2 impossible date test."""

    def test_valid_date_passes(self):
        df = make_profile([5.0, 6.0])
        flags = run_item(QCImpossibleDate, df)
        assert flags["qc_impossible_date"].to_list() == [1, 1]

    @pytest.mark.parametrize(
        "timestamp",
        [
            datetime(1950, 6, 1),  # year not greater than 1950
            datetime(1893, 1, 1),  # far past
            datetime(2100, 1, 1),  # future
            None,  # unparseable/absent date
        ],
    )
    def test_bad_dates_fail(self, timestamp):
        df = make_profile([5.0, 6.0], timestamp=timestamp)
        flags = run_item(QCImpossibleDate, df)
        assert flags["qc_impossible_date"].to_list() == [4, 4]

    def test_min_year_param_override(self):
        df = make_profile([5.0], timestamp=datetime(1960, 1, 1))
        flags = run_item(QCImpossibleDate, df, {"params": {"min_year": 1990}})
        assert flags["qc_impossible_date"].to_list() == [4]

    def test_fail_flag_override(self):
        df = make_profile([5.0], timestamp=None)
        flags = run_item(QCImpossibleDate, df, {"fail_flag": 3})
        assert flags["qc_impossible_date"].to_list() == [3]


# ---------------------------------------------------------------------------
# RTQC3: impossible location
# ---------------------------------------------------------------------------


class TestImpossibleLocation:
    """RTQC3 impossible location test."""

    def test_valid_location_passes(self):
        df = make_profile([5.0], longitude=-180.0, latitude=90.0)
        flags = run_item(QCImpossibleLocation, df)
        assert flags["qc_impossible_location"].to_list() == [1]

    @pytest.mark.parametrize(
        "longitude, latitude",
        [
            (15.0, 90.5),  # latitude too high
            (15.0, -91.0),  # latitude too low
            (180.5, 55.0),  # longitude too high
            (-181.0, 55.0),  # longitude too low
            (None, 55.0),  # missing longitude
            (15.0, None),  # missing latitude
        ],
    )
    def test_bad_locations_fail(self, longitude, latitude):
        df = make_profile([5.0], longitude=longitude, latitude=latitude)
        flags = run_item(QCImpossibleLocation, df)
        assert flags["qc_impossible_location"].to_list() == [4]


# ---------------------------------------------------------------------------
# RTQC6/RTQC7: global and regional range
# ---------------------------------------------------------------------------


class TestGlobalRange:
    """RTQC6 global range test."""

    def test_defaults_flag_out_of_range_values(self):
        df = make_profile(
            temp=[5.0, -3.0, 41.0, None],
            psal=[35.0, 1.9, 41.5, 35.0],
        )
        flags = run_item(QCGlobalRange, df)
        # Null temp passes: a missing value cannot be range-checked.
        assert flags["temp_qc_global_range"].to_list() == [1, 4, 4, 1]
        assert flags["psal_qc_global_range"].to_list() == [1, 4, 4, 1]

    def test_boundary_values_pass(self):
        df = make_profile(temp=[-2.5, 40.0], psal=[2.0, 41.0])
        flags = run_item(QCGlobalRange, df)
        assert flags["temp_qc_global_range"].to_list() == [1, 1]
        assert flags["psal_qc_global_range"].to_list() == [1, 1]

    def test_param_override_per_variable(self):
        """Overriding temp bounds keeps the psal defaults intact."""
        df = make_profile(temp=[5.0], psal=[1.0])
        flags = run_item(
            QCGlobalRange,
            df,
            {"params": {"temp": {"min": 10.0, "max": 20.0}}},
        )
        assert flags["temp_qc_global_range"].to_list() == [4]
        assert flags["psal_qc_global_range"].to_list() == [4]

    def test_col_names_limit_variables(self):
        """An explicit col_names list produces only those flag columns."""
        df = make_profile(temp=[5.0])
        flags = run_item(QCGlobalRange, df, {"col_names": ["temp"]})
        assert "temp_qc_global_range" in flags.columns
        assert "psal_qc_global_range" not in flags.columns

    def test_incomplete_bounds_raise(self):
        df = make_profile(temp=[5.0])
        with pytest.raises(ValueError, match="min"):
            run_item(
                QCGlobalRange,
                df,
                {"col_names": ["temp"], "params": {"temp": {"min": 0.0}}},
            )


class TestRegionalRange:
    """RTQC7 regional range test (per-region config supplies the ranges)."""

    def test_regional_bounds(self):
        """Mediterranean ranges: temp below 10 fails regionally."""
        df = make_profile(temp=[5.0, 15.0], psal=[38.5, 40.5])
        flags = run_item(
            QCRegionalRange,
            df,
            {
                "params": {
                    "temp": {"min": 10.0, "max": 40.0},
                    "psal": {"min": 2.0, "max": 40.0},
                }
            },
        )
        assert flags["temp_qc_regional_range"].to_list() == [4, 1]
        assert flags["psal_qc_regional_range"].to_list() == [1, 4]

    def test_no_params_raise(self):
        """Without configured ranges the item refuses to run (no silent pass)."""
        df = make_profile(temp=[5.0])
        with pytest.raises(ValueError, match="no variables"):
            run_item(QCRegionalRange, df)


# ---------------------------------------------------------------------------
# RTQC13: stuck value
# ---------------------------------------------------------------------------


class TestStuckValue:
    """RTQC13 stuck value test."""

    def test_stuck_profile_fails_entirely(self):
        df = make_profile(temp=[7.7, 7.7, 7.7], psal=[35.0, 35.1, 35.2])
        flags = run_item(QCStuckValue, df)
        assert flags["temp_qc_stuck_value"].to_list() == [4, 4, 4]
        assert flags["psal_qc_stuck_value"].to_list() == [1, 1, 1]

    def test_single_observation_exempt(self):
        df = make_profile(temp=[7.7])
        flags = run_item(QCStuckValue, df)
        assert flags["temp_qc_stuck_value"].to_list() == [1]

    def test_all_null_exempt(self):
        """A variable that was never measured is not 'stuck'."""
        df = make_profile(temp=[None, None, None])
        flags = run_item(QCStuckValue, df)
        assert flags["temp_qc_stuck_value"].to_list() == [1, 1, 1]

    def test_identical_with_nulls_fails(self):
        """Nulls among identical measurements do not mask a stuck sensor."""
        df = make_profile(temp=[7.7, None, 7.7])
        flags = run_item(QCStuckValue, df)
        assert flags["temp_qc_stuck_value"].to_list() == [4, 4, 4]

    def test_profiles_evaluated_independently(self):
        df = pl.concat(
            [
                make_profile(temp=[7.7, 7.7], profile_no=1),
                make_profile(temp=[7.7, 8.0], profile_no=2),
            ]
        )
        flags = run_item(QCStuckValue, df)
        assert flags["temp_qc_stuck_value"].to_list() == [4, 4, 1, 1]

    def test_min_observations_param(self):
        """Raising min_observations exempts short stuck profiles."""
        df = make_profile(temp=[7.7, 7.7])
        flags = run_item(QCStuckValue, df, {"params": {"min_observations": 3}})
        assert flags["temp_qc_stuck_value"].to_list() == [1, 1]


# ---------------------------------------------------------------------------
# Shared plumbing
# ---------------------------------------------------------------------------


class TestQCItemPlumbing:
    """QCItemFeatureBase output modes, dtypes, and registry entries."""

    def test_full_frame_output(self):
        """Without selected_rows, all rows are returned with key columns."""
        df = make_profile(temp=[5.0, -3.0])
        flags = run_item(QCGlobalRange, df)
        assert flags.shape[0] == 2
        assert flags.columns == [
            "platform_code",
            "profile_no",
            "observation_no",
            "temp_qc_global_range",
            "psal_qc_global_range",
        ]

    def test_selected_rows_output(self):
        """With selected_rows, features are row_id-keyed for those rows."""
        df = make_profile(temp=[5.0, -3.0, 41.0])
        selected = pl.DataFrame(
            {
                "row_id": [10, 11],
                "platform_code": ["P1", "P1"],
                "profile_no": [1, 1],
                "observation_no": [3, 1],  # deliberately reordered
            }
        )
        ds = QCGlobalRange(
            target_name="temp",
            filtered_input=df,
            selected_rows={"temp": selected},
        )
        ds.extract_features()

        assert ds.features.columns == [
            "row_id",
            "temp_qc_global_range",
            "psal_qc_global_range",
        ]
        # Order follows selected_rows: observation 3 (temp 41 -> 4), then 1.
        assert ds.features["row_id"].to_list() == [10, 11]
        assert ds.features["temp_qc_global_range"].to_list() == [4, 1]

    def test_flags_never_null_and_int64(self):
        df = make_profile(temp=[None, 5.0])
        flags = run_item(QCGlobalRange, df)
        assert flags["temp_qc_global_range"].null_count() == 0
        assert flags["temp_qc_global_range"].dtype == pl.Int64

    def test_registry_entries(self):
        """All Phase 3 items are registered under qc_-prefixed names."""
        expected = {
            "qc_impossible_date": QCImpossibleDate,
            "qc_impossible_location": QCImpossibleLocation,
            "qc_global_range": QCGlobalRange,
            "qc_regional_range": QCRegionalRange,
            "qc_stuck_value": QCStuckValue,
        }
        for name, cls in expected.items():
            assert FEATURE_REGISTRY[name] is cls
