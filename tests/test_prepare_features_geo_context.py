"""Unit tests for the geo_context feature class.

The bathymetry and coast distance columns are inputs, not lookups, so the
tests are about what the class does with them: the sign convention, the
two ways of expressing where in the water column a level sits, and what
happens when the dataset does not carry them at all.
"""

import polars as pl
import pytest

from aiqclib.prepare.features.geo_context import GeoContext

from tests.test_prepare_features_derived import make_profile, run_feature

# 10, 20 and 30 dbar, so roughly 10, 20 and 30 metres down.
SHELF = {"bathymetry": [60.0, 60.0, 60.0]}


def shelf_profile(extra=None, temp=None):
    """A three-level profile over a 60 metre sea floor."""
    values = dict(SHELF)
    values.update(extra or {})
    return make_profile(temp or [10.0, 9.0, 8.0], extra=values)


class TestOutputs:
    """What the class emits."""

    def test_outputs_selects_a_subset(self):
        out = run_feature(
            GeoContext, shelf_profile(), {"outputs": ["normalized_depth"]}
        )
        assert out.columns == [
            "platform_code",
            "profile_no",
            "observation_no",
            "normalized_depth",
        ]

    def test_nothing_is_per_variable(self):
        out = run_feature(GeoContext, shelf_profile(), {"outputs": ["bathymetry"]})
        assert "bathymetry" in out.columns
        assert "temp_bathymetry" not in out.columns

    def test_unknown_output_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown output"):
            run_feature(GeoContext, shelf_profile(), {"outputs": ["ocean_basin"]})


class TestWaterColumnPosition:
    """The two ways of saying where in the water column a level is."""

    def test_normalized_depth_runs_from_surface_to_bottom(self):
        out = run_feature(
            GeoContext, shelf_profile(), {"outputs": ["normalized_depth"]}
        )
        values = out["normalized_depth"].to_list()
        assert all(0.0 < value < 1.0 for value in values)
        assert values[0] < values[1] < values[2]

    def test_normalized_depth_makes_two_places_comparable(self):
        """Half way down is 0.5 over a shelf and over a basin alike."""
        shallow = run_feature(
            GeoContext,
            make_profile([10.0], pres=[50.0], extra={"bathymetry": [100.0]}),
            {"outputs": ["normalized_depth"]},
        )
        deep = run_feature(
            GeoContext,
            make_profile([10.0], pres=[1000.0], extra={"bathymetry": [1990.0]}),
            {"outputs": ["normalized_depth"]},
        )
        assert shallow["normalized_depth"][0] == pytest.approx(0.5, abs=0.02)
        assert deep["normalized_depth"][0] == pytest.approx(0.5, abs=0.02)

    def test_distance_to_bottom_is_in_metres(self):
        out = run_feature(
            GeoContext, shelf_profile(), {"outputs": ["distance_to_bottom"]}
        )
        # 10, 20 and 30 dbar below the surface of 60 metres of water. A
        # decibar is slightly more than a metre, so the answers sit a
        # little above the round numbers.
        assert out["distance_to_bottom"].to_list() == pytest.approx(
            [50.0, 40.0, 30.0], abs=0.5
        )

    def test_sea_level_and_above_is_not_a_sea_floor(self):
        out = run_feature(
            GeoContext,
            shelf_profile({"bathymetry": [0.0, -5.0, 60.0]}),
            {"outputs": ["bathymetry"]},
        )
        assert out["bathymetry"].to_list()[:2] == [None, None]
        assert out["bathymetry"].to_list()[2] == 60.0


class TestSignConvention:
    """Which sign means deeper is a parameter, because both are used."""

    def test_negative_convention_is_read_correctly(self):
        out = run_feature(
            GeoContext,
            shelf_profile({"bathymetry": [-60.0, -60.0, -60.0]}),
            {"outputs": ["bathymetry"], "params": {"positive_depth": False}},
        )
        assert out["bathymetry"].to_list() == [60.0, 60.0, 60.0]

    def test_reading_the_wrong_convention_yields_nothing(self):
        """Getting it wrong gives null, not a silently inverted answer."""
        out = run_feature(
            GeoContext,
            shelf_profile({"bathymetry": [-60.0, -60.0, -60.0]}),
            {"outputs": ["bathymetry"]},
        )
        assert out["bathymetry"].to_list() == [None, None, None]


class TestDeepStableLayer:
    """The deep, weakly stratified flag."""

    def test_shallow_levels_are_not_flagged(self):
        out = run_feature(
            GeoContext, shelf_profile(), {"outputs": ["deep_stable_layer"]}
        )
        assert out["deep_stable_layer"].to_list()[1] == 0

    def test_deep_and_uniform_is_flagged(self):
        deep = make_profile(
            [4.0, 4.0, 4.0, 4.0, 4.0],
            pres=[1500.0, 1600.0, 1700.0, 1800.0, 1900.0],
            extra={"bathymetry": [3000.0] * 5},
        )
        out = run_feature(GeoContext, deep, {"outputs": ["deep_stable_layer"]})
        assert out["deep_stable_layer"].to_list()[1:4] == [1, 1, 1]

    def test_deep_but_sharply_stratified_is_not_flagged(self):
        deep = make_profile(
            [4.0, 4.0, 14.0, 4.0, 4.0],
            pres=[1500.0, 1600.0, 1700.0, 1800.0, 1900.0],
            extra={"bathymetry": [3000.0] * 5},
        )
        out = run_feature(GeoContext, deep, {"outputs": ["deep_stable_layer"]})
        assert 0 in out["deep_stable_layer"].to_list()[1:4]

    def test_threshold_is_configurable(self):
        """Lowering the depth lets a uniform shelf column qualify."""
        uniform = shelf_profile(temp=[8.0, 8.0, 8.0])
        assert (
            run_feature(GeoContext, uniform, {"outputs": ["deep_stable_layer"]})[
                "deep_stable_layer"
            ].to_list()[1]
            == 0
        )
        out = run_feature(
            GeoContext,
            uniform,
            {"outputs": ["deep_stable_layer"], "params": {"deep_threshold": 5.0}},
        )
        assert out["deep_stable_layer"].to_list()[1] == 1

    def test_profile_ends_are_null(self):
        out = run_feature(
            GeoContext, shelf_profile(), {"outputs": ["deep_stable_layer"]}
        )
        values = out["deep_stable_layer"].to_list()
        assert values[0] is None and values[-1] is None


class TestMissingColumns:
    """What happens when the dataset does not carry the grids."""

    def test_missing_bathymetry_is_an_error_by_default(self):
        df = shelf_profile().drop("bathymetry")
        with pytest.raises(ValueError, match="needs the column"):
            run_feature(GeoContext, df, {"outputs": ["normalized_depth"]})

    def test_required_false_emits_nulls_with_a_warning(self):
        df = shelf_profile().drop("bathymetry")
        with pytest.warns(UserWarning, match="required"):
            out = run_feature(
                GeoContext,
                df,
                {"outputs": ["normalized_depth"], "params": {"required": False}},
            )
        assert out["normalized_depth"].to_list() == [None, None, None]

    def test_the_schema_is_the_same_either_way(self):
        """One configuration over several regions keeps one frame shape."""
        with_column = run_feature(
            GeoContext, shelf_profile(), {"outputs": ["normalized_depth"]}
        )
        with pytest.warns(UserWarning):
            without = run_feature(
                GeoContext,
                shelf_profile().drop("bathymetry"),
                {"outputs": ["normalized_depth"], "params": {"required": False}},
            )
        assert with_column.columns == without.columns

    def test_coast_distance_is_passed_through(self):
        df = shelf_profile({"coast_distance": [12.0, 12.0, 12.0]})
        out = run_feature(GeoContext, df, {"outputs": ["coast_distance"]})
        assert out["coast_distance"].to_list() == [12.0, 12.0, 12.0]

    def test_column_names_are_configurable(self):
        df = shelf_profile().rename({"bathymetry": "seafloor_m"})
        out = run_feature(
            GeoContext,
            df,
            {
                "outputs": ["bathymetry"],
                "params": {"bathymetry_column": "seafloor_m"},
            },
        )
        assert out["bathymetry"].to_list() == [60.0, 60.0, 60.0]

    def test_outputs_that_need_nothing_extra_still_work(self):
        """coast_distance alone must not demand a bathymetry column."""
        df = shelf_profile({"coast_distance": [1.0, 2.0, 3.0]}).drop("bathymetry")
        out = run_feature(GeoContext, df, {"outputs": ["coast_distance"]})
        assert out["coast_distance"].to_list() == [1.0, 2.0, 3.0]


class TestProfileIsolation:
    """Profiles stay independent."""

    def test_two_profiles_do_not_interact(self):
        first = shelf_profile()
        second = shelf_profile().with_columns(
            pl.lit(2, dtype=pl.Int64).alias("profile_no")
        )
        both = run_feature(
            GeoContext, first.vstack(second), {"outputs": ["deep_stable_layer"]}
        )
        assert both["deep_stable_layer"].to_list() == [None, 0, None, None, 0, None]
