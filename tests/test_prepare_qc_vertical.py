"""Unit tests for the vertical-profile NRT QC item feature classes.

Covers the Phase 4 items (pressure increasing, spike, gradient, digit
rollover). All expectations are hand-computed row by row on crafted
profiles, including the depth-threshold switch at 500 db, profile
boundaries, null neighbours, and out-of-order input rows (the stencils
follow observation order, not row order).
"""

from datetime import datetime

import polars as pl
import pytest

from aiqclib.common.loader.feature_registry import FEATURE_REGISTRY
from aiqclib.prepare.features.qc_digit_rollover import QCDigitRollover
from aiqclib.prepare.features.qc_gradient import QCGradient
from aiqclib.prepare.features.qc_pressure_increasing import QCPressureIncreasing
from aiqclib.prepare.features.qc_spike import QCSpike

VALID_TS = datetime(2023, 6, 1, 12, 0)


def make_profile(
    temp: list,
    psal: list = None,
    pres: list = None,
    platform_code: str = "P1",
    profile_no: int = 1,
) -> pl.DataFrame:
    """Build one synthetic profile with the standard input columns."""
    n = len(temp)
    psal = psal if psal is not None else [35.0] * n
    pres = pres if pres is not None else [float(10 * i) for i in range(1, n + 1)]
    return pl.DataFrame(
        {
            "platform_code": [platform_code] * n,
            "profile_no": [profile_no] * n,
            "observation_no": list(range(1, n + 1)),
            "profile_timestamp": pl.Series([VALID_TS] * n, dtype=pl.Datetime("ms")),
            "longitude": [15.0] * n,
            "latitude": [55.0] * n,
            "pres": pres,
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
# RTQC8: pressure increasing
# ---------------------------------------------------------------------------


class TestPressureIncreasing:
    """RTQC8 pressure increasing test."""

    def test_monotonic_profile_passes(self):
        df = make_profile(temp=[5.0] * 4, pres=[10.0, 20.0, 30.0, 40.0])
        flags = run_item(QCPressureIncreasing, df)
        assert flags["qc_pressure_increasing"].to_list() == [1, 1, 1, 1]

    def test_constant_run_flags_all_but_first(self):
        df = make_profile(temp=[5.0] * 5, pres=[10.0, 20.0, 20.0, 20.0, 30.0])
        flags = run_item(QCPressureIncreasing, df)
        assert flags["qc_pressure_increasing"].to_list() == [1, 1, 4, 4, 1]

    def test_reversed_segment_flagged_until_recovery(self):
        """Pressures below the running maximum are the reversed part."""
        df = make_profile(temp=[5.0] * 6, pres=[10.0, 20.0, 30.0, 25.0, 28.0, 35.0])
        flags = run_item(QCPressureIncreasing, df)
        assert flags["qc_pressure_increasing"].to_list() == [1, 1, 1, 4, 4, 1]

    def test_profiles_independent(self):
        """The last pressure of one profile does not constrain the next."""
        df = pl.concat(
            [
                make_profile(temp=[5.0] * 2, pres=[10.0, 20.0], profile_no=1),
                make_profile(temp=[5.0] * 2, pres=[5.0, 15.0], profile_no=2),
            ]
        )
        flags = run_item(QCPressureIncreasing, df)
        assert flags["qc_pressure_increasing"].to_list() == [1, 1, 1, 1]

    def test_out_of_order_rows(self):
        """Flags follow observation order even when rows are shuffled."""
        df = make_profile(temp=[5.0] * 4, pres=[10.0, 20.0, 15.0, 30.0]).sample(
            fraction=1.0, shuffle=True, seed=42
        )
        flags = run_item(QCPressureIncreasing, df).sort("observation_no")
        assert flags["qc_pressure_increasing"].to_list() == [1, 1, 4, 1]


# ---------------------------------------------------------------------------
# RTQC9: spike
# ---------------------------------------------------------------------------


class TestSpike:
    """RTQC9 spike test."""

    def test_shallow_temperature_spike(self):
        """|17-10| - 0 = 7 > 6 (shallow) flags only the middle value."""
        df = make_profile(temp=[10.0, 17.0, 10.0], pres=[100.0, 110.0, 120.0])
        flags = run_item(QCSpike, df)
        assert flags["temp_qc_spike"].to_list() == [1, 4, 1]

    def test_below_threshold_passes(self):
        """Test value 5.9 stays under the shallow threshold of 6."""
        df = make_profile(temp=[10.0, 15.9, 10.0], pres=[100.0, 110.0, 120.0])
        flags = run_item(QCSpike, df)
        assert flags["temp_qc_spike"].to_list() == [1, 1, 1]

    def test_deep_threshold_applies(self):
        """The same excursion passes shallow (3 < 6) but fails deep (3 > 2)."""
        shallow = make_profile(temp=[5.0, 8.0, 5.0], pres=[100.0, 110.0, 120.0])
        deep = make_profile(temp=[5.0, 8.0, 5.0], pres=[600.0, 610.0, 620.0])
        assert run_item(QCSpike, shallow)["temp_qc_spike"].to_list() == [1, 1, 1]
        assert run_item(QCSpike, deep)["temp_qc_spike"].to_list() == [1, 4, 1]

    def test_steep_gradient_is_not_a_spike(self):
        """A steady steep slope has spike test value <= 0."""
        df = make_profile(temp=[10.0, 13.0, 16.0], pres=[100.0, 110.0, 120.0])
        flags = run_item(QCSpike, df)
        assert flags["temp_qc_spike"].to_list() == [1, 1, 1]

    def test_salinity_spike(self):
        """|36-35| - 0 = 1 > 0.9 (shallow) flags the salinity value."""
        df = make_profile(
            temp=[10.0] * 3,
            psal=[35.0, 36.0, 35.0],
            pres=[100.0, 110.0, 120.0],
        )
        flags = run_item(QCSpike, df)
        assert flags["psal_qc_spike"].to_list() == [1, 4, 1]
        assert flags["temp_qc_spike"].to_list() == [1, 1, 1]

    def test_null_neighbour_passes(self):
        """A null in the stencil makes the value untestable, not bad."""
        df = make_profile(
            temp=[10.0, 17.0, None, 10.0],
            pres=[100.0, 110.0, 120.0, 130.0],
        )
        flags = run_item(QCSpike, df)
        assert flags["temp_qc_spike"].to_list() == [1, 1, 1, 1]

    def test_out_of_order_rows(self):
        """The stencil follows observation order, not row order."""
        df = make_profile(temp=[10.0, 17.0, 10.0], pres=[100.0, 110.0, 120.0]).sample(
            fraction=1.0, shuffle=True, seed=7
        )
        flags = run_item(QCSpike, df).sort("observation_no")
        assert flags["temp_qc_spike"].to_list() == [1, 4, 1]


# ---------------------------------------------------------------------------
# RTQC11: gradient
# ---------------------------------------------------------------------------


class TestGradient:
    """RTQC11 gradient test."""

    def test_linear_profile_passes(self):
        """On a linear slope V2 equals the neighbour mean: test value 0."""
        df = make_profile(temp=[10.0, 13.0, 16.0], pres=[100.0, 110.0, 120.0])
        flags = run_item(QCGradient, df)
        assert flags["temp_qc_gradient"].to_list() == [1, 1, 1]

    def test_shallow_temperature_gradient(self):
        """|12 - 2| = 10 > 9 (shallow) flags the middle value."""
        df = make_profile(temp=[2.0, 12.0, 2.0], pres=[100.0, 110.0, 120.0])
        flags = run_item(QCGradient, df)
        assert flags["temp_qc_gradient"].to_list() == [1, 4, 1]

    def test_gradient_fails_where_spike_passes(self):
        """V1=0, V2=14, V3=8: gradient 10 > 9 fails, spike 6 <= 6 passes."""
        df = make_profile(temp=[0.0, 14.0, 8.0], pres=[100.0, 110.0, 120.0])
        assert run_item(QCGradient, df)["temp_qc_gradient"].to_list() == [1, 4, 1]
        assert run_item(QCSpike, df)["temp_qc_spike"].to_list() == [1, 1, 1]

    def test_deep_threshold_applies(self):
        """|9-5| = 4: passes shallow (4 < 9) but fails deep (4 > 3)."""
        shallow = make_profile(temp=[5.0, 9.0, 5.0], pres=[100.0, 110.0, 120.0])
        deep = make_profile(temp=[5.0, 9.0, 5.0], pres=[600.0, 610.0, 620.0])
        assert run_item(QCGradient, shallow)["temp_qc_gradient"].to_list() == [
            1,
            1,
            1,
        ]
        assert run_item(QCGradient, deep)["temp_qc_gradient"].to_list() == [
            1,
            4,
            1,
        ]

    def test_salinity_gradient(self):
        """|36.6 - 35| = 1.6 > 1.5 (shallow) flags the salinity value."""
        df = make_profile(
            temp=[10.0] * 3,
            psal=[35.0, 36.6, 35.0],
            pres=[100.0, 110.0, 120.0],
        )
        flags = run_item(QCGradient, df)
        assert flags["psal_qc_gradient"].to_list() == [1, 4, 1]

    def test_missing_thresholds_raise(self):
        df = make_profile(temp=[5.0])
        with pytest.raises(ValueError, match="shallow"):
            run_item(
                QCGradient,
                df,
                {"col_names": ["temp"], "params": {"temp": {"shallow": 9.0}}},
            )


# ---------------------------------------------------------------------------
# RTQC12: digit rollover
# ---------------------------------------------------------------------------


class TestDigitRollover:
    """RTQC12 digit rollover test."""

    def test_temperature_jump_flagged(self):
        """|15.5 - 5.0| = 10.5 > 10 flags the second value only."""
        df = make_profile(temp=[5.0, 15.5, 15.0])
        flags = run_item(QCDigitRollover, df)
        assert flags["temp_qc_digit_rollover"].to_list() == [1, 4, 1]

    def test_threshold_is_exclusive(self):
        """A difference equal to the threshold passes (not greater than)."""
        df = make_profile(temp=[5.0, 15.0])
        flags = run_item(QCDigitRollover, df)
        assert flags["temp_qc_digit_rollover"].to_list() == [1, 1]

    def test_salinity_jump_flagged(self):
        """|29.4 - 35.0| = 5.6 > 5 flags the salinity value."""
        df = make_profile(temp=[5.0, 5.0], psal=[35.0, 29.4])
        flags = run_item(QCDigitRollover, df)
        assert flags["psal_qc_digit_rollover"].to_list() == [1, 4]
        assert flags["temp_qc_digit_rollover"].to_list() == [1, 1]

    def test_profile_boundary_not_compared(self):
        """The first observation of a profile has no previous value."""
        df = pl.concat(
            [
                make_profile(temp=[5.0, 6.0], profile_no=1),
                make_profile(temp=[25.0, 26.0], profile_no=2),
            ]
        )
        flags = run_item(QCDigitRollover, df)
        assert flags["temp_qc_digit_rollover"].to_list() == [1, 1, 1, 1]

    def test_param_override(self):
        df = make_profile(temp=[5.0, 8.0])
        flags = run_item(QCDigitRollover, df, {"params": {"temp": 2.0}})
        assert flags["temp_qc_digit_rollover"].to_list() == [1, 4]

    def test_non_numeric_threshold_raises(self):
        df = make_profile(temp=[5.0])
        with pytest.raises(ValueError, match="numeric"):
            run_item(
                QCDigitRollover,
                df,
                {"col_names": ["temp"], "params": {"temp": {"max": 10.0}}},
            )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistryEntries:
    """All Phase 4 items are registered under qc_-prefixed names."""

    def test_registry_entries(self):
        expected = {
            "qc_pressure_increasing": QCPressureIncreasing,
            "qc_spike": QCSpike,
            "qc_gradient": QCGradient,
            "qc_digit_rollover": QCDigitRollover,
        }
        for name, cls in expected.items():
            assert FEATURE_REGISTRY[name] is cls
