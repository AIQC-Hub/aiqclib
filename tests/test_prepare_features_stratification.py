"""Unit tests for the stratification feature class.

The profiles here are built so the physics is obvious by inspection: a
cooling profile is stable (density rises with depth), a warming one is
inverted, and an isothermal, isohaline one is neutral apart from the
pressure effect the potential density anomaly removes.
"""

import polars as pl
import pytest

from tests.test_prepare_features_derived import make_profile, run_feature

from aiqclib.prepare.features.stratification import Stratification


class TestOutputs:
    """What the class emits."""

    def test_all_outputs_by_default(self):
        out = run_feature(Stratification, make_profile([20.0, 15.0, 10.0]))
        assert set(out.columns) == {
            "platform_code",
            "profile_no",
            "observation_no",
            "sigma0_gradient",
            "n2",
            "n2_abs",
            "unstable_flag",
        }

    def test_outputs_selects_a_subset(self):
        out = run_feature(
            Stratification, make_profile([20.0, 15.0, 10.0]), {"outputs": ["n2"]}
        )
        assert "n2" in out.columns
        assert "sigma0_gradient" not in out.columns

    def test_unknown_output_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown output"):
            run_feature(Stratification, make_profile([10.0]), {"outputs": ["buoyancy"]})


class TestStability:
    """The sign of the answer, which is the whole point of the feature."""

    def test_cooling_profile_is_stable(self):
        """Colder water below is denser water below."""
        out = run_feature(Stratification, make_profile([20.0, 15.0, 10.0]))
        assert out["n2"][1] > 0.0
        assert out["unstable_flag"].to_list() == [None, 0, None]

    def test_warming_profile_is_unstable(self):
        out = run_feature(Stratification, make_profile([10.0, 15.0, 20.0]))
        assert out["n2"][1] < 0.0
        assert out["unstable_flag"].to_list() == [None, 1, None]

    def test_magnitude_ignores_the_direction(self):
        """Reversing the profile gives the same strength, opposite sign.

        Only nearly the same strength: the two profiles put their warm
        water at different pressures, and the equation of state is not
        linear, so the potential densities are not exact mirrors.
        """
        stable = run_feature(Stratification, make_profile([20.0, 15.0, 10.0]))
        unstable = run_feature(Stratification, make_profile([10.0, 15.0, 20.0]))
        assert stable["n2"][1] == pytest.approx(-unstable["n2"][1], rel=1e-3)
        assert stable["n2_abs"][1] == pytest.approx(unstable["n2_abs"][1], rel=1e-3)
        assert stable["n2_abs"][1] > 0.0

    def test_uniform_profile_is_all_but_neutral(self):
        """A column of constant in-situ temperature is barely stratified.

        Not exactly neutral: sigma-0 is referenced to the surface, and a
        parcel lifted from depth cools adiabatically, so constant in-situ
        temperature hides a potential temperature that falls with depth.
        The residual stratification is four orders of magnitude below a
        thermocline, which is the point being pinned here.
        """
        out = run_feature(Stratification, make_profile([10.0] * 5))
        stratified = run_feature(Stratification, make_profile([20.0, 15.0, 10.0]))
        assert 0.0 < out["n2"][2] < 1e-6
        assert out["n2"][2] < stratified["n2"][1] / 1000.0

    def test_gradient_has_the_same_sign_as_n2(self):
        out = run_feature(Stratification, make_profile([20.0, 15.0, 10.0]))
        assert out["sigma0_gradient"][1] > 0.0

    def test_threshold_is_configurable(self):
        """A weakly stratified level can be called unstable if wanted."""
        profile = make_profile([20.0, 15.0, 10.0])
        strict = run_feature(Stratification, profile, {"params": {"unstable_n2": 1.0}})
        assert strict["unstable_flag"].to_list() == [None, 1, None]


class TestEdgesAndMissingValues:
    """Where the answer is null, and why."""

    def test_profile_ends_are_null(self):
        """The first and last level have no pair of neighbours."""
        out = run_feature(Stratification, make_profile([20.0, 15.0, 10.0, 5.0]))
        assert out["n2"].to_list()[0] is None
        assert out["n2"].to_list()[-1] is None

    def test_profiles_do_not_reach_into_each_other(self):
        first = make_profile([20.0, 15.0, 10.0])
        second = make_profile([20.0, 15.0, 10.0]).with_columns(
            pl.lit(2, dtype=pl.Int64).alias("profile_no")
        )
        out = run_feature(Stratification, first.vstack(second))
        assert out["n2"].to_list()[2] is None
        assert out["n2"].to_list()[3] is None

    def test_placeholder_temperature_gives_null(self):
        out = run_feature(Stratification, make_profile([20.0, -999.0, 10.0, 5.0]))
        assert out["n2"].to_list()[1] is None

    def test_levels_at_the_same_depth_give_null_not_infinity(self):
        out = run_feature(
            Stratification,
            make_profile([20.0, 15.0, 10.0], pres=[10.0, 10.0, 10.0]),
        )
        assert out["sigma0_gradient"].to_list()[1] is None
        assert out["n2"].to_list()[1] is None

    def test_unstable_flag_is_null_where_n2_is(self):
        """An unknown stability is not a stable one."""
        out = run_feature(Stratification, make_profile([20.0, -999.0, 10.0, 5.0]))
        assert out["unstable_flag"].to_list()[1] is None


class TestConfiguration:
    """Input columns."""

    def test_existing_sigma0_column_is_used(self):
        """Density supplied upstream is trusted rather than recomputed."""
        profile = make_profile([20.0, 15.0, 10.0], extra={"dens": [20.0, 25.0, 30.0]})
        out = run_feature(
            Stratification, profile, {"params": {"sigma0_column": "dens"}}
        )
        assert out["sigma0_gradient"][1] > 0.0

    def test_missing_input_column_is_an_error(self):
        profile = make_profile([20.0, 15.0, 10.0]).drop("latitude")
        with pytest.raises(ValueError, match="needs the column"):
            run_feature(Stratification, profile)
