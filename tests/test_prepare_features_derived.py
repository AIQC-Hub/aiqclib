"""Unit tests for the derived_values feature class.

The expected sigma-0 values are the ones already pinned by the density
inversion tests: at psal 35, temperatures [10, 20, 12] give sigma-0
[26.953, 24.764, 26.590].
"""

from datetime import datetime

import polars as pl
import pytest

from aiqclib.prepare.features.derived_values import DerivedValues

VALID_TS = datetime(2023, 6, 1, 12, 0)


def make_profile(
    temp: list,
    psal: list = None,
    pres: list = None,
    latitude: float = 55.0,
    extra: dict = None,
) -> pl.DataFrame:
    """Build one synthetic profile with the standard input columns."""
    n = len(temp)
    psal = psal if psal is not None else [35.0] * n
    pres = pres if pres is not None else [float(10 * i) for i in range(1, n + 1)]
    df = pl.DataFrame(
        {
            "platform_code": ["P1"] * n,
            "profile_no": [1] * n,
            "observation_no": list(range(1, n + 1)),
            "profile_timestamp": pl.Series([VALID_TS] * n, dtype=pl.Datetime("ms")),
            "longitude": [15.0] * n,
            "latitude": [latitude] * n,
            "pres": pres,
            "temp": temp,
            "psal": psal,
        }
    )
    if extra:
        df = df.with_columns(
            [pl.Series(name, values) for name, values in extra.items()]
        )
    return df


def run_feature(cls, df: pl.DataFrame, feature_info: dict = None) -> pl.DataFrame:
    """Run a feature in full-frame mode (no selected rows) and return it."""
    ds = cls(feature_info=feature_info or {}, filtered_input=df)
    ds.extract_features()
    return ds.features


class TestOutputs:
    """What the class emits, and how the outputs entry narrows it."""

    def test_all_outputs_by_default(self):
        out = run_feature(DerivedValues, make_profile([10.0, 20.0, 12.0]))
        assert set(out.columns) == {
            "platform_code",
            "profile_no",
            "observation_no",
            "sigma0",
            "depth",
            "potential_temperature",
        }

    def test_outputs_selects_a_subset(self):
        out = run_feature(
            DerivedValues, make_profile([10.0, 20.0]), {"outputs": ["sigma0"]}
        )
        assert "sigma0" in out.columns
        assert "depth" not in out.columns

    def test_unknown_output_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown output"):
            run_feature(DerivedValues, make_profile([10.0]), {"outputs": ["salinity"]})

    def test_empty_outputs_is_rejected(self):
        with pytest.raises(ValueError, match="empty 'outputs' list"):
            run_feature(DerivedValues, make_profile([10.0]), {"outputs": []})


class TestValues:
    """The numbers themselves."""

    def test_sigma0_matches_the_equation_of_state(self):
        out = run_feature(DerivedValues, make_profile([10.0, 20.0, 12.0]))
        assert out["sigma0"].to_list() == pytest.approx(
            [26.953, 24.764, 26.590], abs=1e-3
        )

    def test_depth_is_slightly_less_than_pressure(self):
        out = run_feature(DerivedValues, make_profile([10.0] * 3))
        depths = out["depth"].to_list()
        pressures = [10.0, 20.0, 30.0]
        assert all(0 < d < p for d, p in zip(depths, pressures))

    def test_potential_temperature_is_below_in_situ_at_depth(self):
        """Lifting a parcel to the surface lets it expand and cool."""
        out = run_feature(DerivedValues, make_profile([10.0], pres=[4000.0]))
        assert out["potential_temperature"][0] < 10.0


class TestMissingAndImpossibleInputs:
    """Nothing is invented for a value the equation of state cannot take."""

    def test_null_temperature_gives_null_outputs(self):
        df = make_profile([10.0, None, 12.0])
        out = run_feature(DerivedValues, df)
        assert out["sigma0"].to_list()[1] is None
        assert out["sigma0"].to_list()[0] is not None

    def test_placeholder_value_gives_null_not_a_number(self):
        """-999 degrees has a finite density, and it is nonsense."""
        out = run_feature(DerivedValues, make_profile([10.0, -999.0, 12.0]))
        assert out["sigma0"].to_list()[1] is None

    def test_depth_survives_a_bad_temperature(self):
        """Depth does not depend on temperature, so it is still computed."""
        out = run_feature(DerivedValues, make_profile([10.0, -999.0, 12.0]))
        assert out["depth"].to_list()[1] is not None


class TestConfiguration:
    """Column names and passthrough."""

    def test_input_columns_are_configurable(self):
        df = make_profile([10.0, 20.0]).rename({"temp": "temperature"})
        out = run_feature(
            DerivedValues,
            df,
            {"outputs": ["sigma0"], "params": {"temperature_column": "temperature"}},
        )
        assert out["sigma0"].to_list() == pytest.approx([26.953, 24.764], abs=1e-3)

    def test_missing_input_column_is_an_error(self):
        df = make_profile([10.0, 20.0]).drop("psal")
        with pytest.raises(ValueError, match="needs the column"):
            run_feature(DerivedValues, df, {"outputs": ["sigma0"]})

    def test_an_existing_column_wins_over_recomputing(self):
        df = make_profile([10.0, 20.0], extra={"sigma0": [1.0, 2.0]})
        out = run_feature(DerivedValues, df, {"outputs": ["sigma0"]})
        assert out["sigma0"].to_list() == [1.0, 2.0]

    def test_passthrough_can_be_turned_off(self):
        df = make_profile([10.0, 20.0], extra={"sigma0": [1.0, 2.0]})
        out = run_feature(
            DerivedValues,
            df,
            {"outputs": ["sigma0"], "params": {"prefer_input_columns": False}},
        )
        assert out["sigma0"].to_list() == pytest.approx([26.953, 24.764], abs=1e-3)

    def test_missing_column_is_not_required_when_passed_through(self):
        """A dataset with its own sigma-0 need not carry salinity at all."""
        df = make_profile([10.0, 20.0], extra={"sigma0": [1.0, 2.0]}).drop("psal")
        out = run_feature(DerivedValues, df, {"outputs": ["sigma0"]})
        assert out["sigma0"].to_list() == [1.0, 2.0]


class TestSelectedRows:
    """The narrowing to a target's selected rows."""

    def test_features_are_keyed_by_row_id(self):
        df = make_profile([10.0, 20.0, 12.0])
        selected = pl.DataFrame(
            {
                "row_id": [101, 102],
                "platform_code": ["P1", "P1"],
                "profile_no": [1, 1],
                "observation_no": [1, 3],
            }
        )
        ds = DerivedValues(
            target_name="temp",
            feature_info={"outputs": ["sigma0"]},
            filtered_input=df,
            selected_rows={"temp": selected},
        )
        ds.extract_features()
        assert ds.features.columns == ["row_id", "sigma0"]
        assert ds.features["row_id"].to_list() == [101, 102]
        assert ds.features["sigma0"].to_list() == pytest.approx(
            [26.953, 26.590], abs=1e-3
        )

    def test_profile_level_rows_are_rejected(self):
        """Observation-level features need an agg list at profile level."""
        df = make_profile([10.0, 20.0])
        selected = pl.DataFrame(
            {"row_id": [1], "platform_code": ["P1"], "profile_no": [1]}
        )
        ds = DerivedValues(
            target_name="temp", filtered_input=df, selected_rows={"temp": selected}
        )
        with pytest.raises(ValueError, match="observation-level"):
            ds.extract_features()
