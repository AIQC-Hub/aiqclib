"""Unit tests for the point-level feature classes.

The profiles here are built so the expected answer is arithmetic rather
than oceanography: a linear ramp with one level displaced by a known
amount, so the smoothed value, the residual and the differences can all be
predicted by hand, and the spike is the only thing any anomaly score
should react to.
"""

import random

import polars as pl
import pytest

from aiqclib.prepare.features.neighbor_diff import NeighborDiff
from aiqclib.prepare.features.profile_smooth import ProfileSmooth
from aiqclib.prepare.features.regime_flags import RegimeFlags

from tests.test_prepare_features_derived import make_profile, run_feature

# A 21-level ramp with a single displaced level in the middle: temperature
# falls by 1 degree per level, and level 11 is 5 degrees too warm.
RAMP = [20.0 - index for index in range(21)]
SPIKED = list(RAMP)
SPIKED[10] += 5.0

# A perfectly linear ramp has identical differences at every level, so its
# median absolute deviation is zero and every robust score is undefined.
# Real profiles are never that clean, so this one carries a small, fixed
# jitter that gives the robust scores a scale to work against.
_JITTER = random.Random(7)
NOISY = [value + _JITTER.uniform(-0.15, 0.15) for value in RAMP]
NOISY_SPIKED = list(NOISY)
NOISY_SPIKED[10] += 5.0


def ramp_profile(values=None, pres=None):
    """A 21-level profile with 10 dbar spacing."""
    values = RAMP if values is None else values
    return make_profile(values, pres=pres)


class TestProfileSmoothOutputs:
    """What the class emits."""

    def test_default_outputs_cover_every_supported_column(self):
        out = run_feature(
            ProfileSmooth,
            ramp_profile(),
            {"col_names": ["temp"], "params": {"windows": [5]}},
        )
        for name in (
            "smooth",
            "d1",
            "d2",
            "residual",
            "robust_z",
            "curvature_ratio",
            "spike_index",
        ):
            assert f"temp_{name}" in out.columns
        assert "temp_w5_outlier_frac" in out.columns
        assert "temp_w5_high_curvature_frac" in out.columns

    def test_outputs_narrow_the_result(self):
        out = run_feature(
            ProfileSmooth,
            ramp_profile(),
            {"col_names": ["temp"], "outputs": ["residual"]},
        )
        assert out.columns == [
            "platform_code",
            "profile_no",
            "observation_no",
            "temp_residual",
        ]

    def test_windows_multiply_only_the_windowed_outputs(self):
        out = run_feature(
            ProfileSmooth,
            ramp_profile(),
            {
                "col_names": ["temp"],
                "outputs": ["residual", "outlier_frac"],
                "params": {"windows": [5, 11]},
            },
        )
        assert "temp_w5_outlier_frac" in out.columns
        assert "temp_w11_outlier_frac" in out.columns
        assert len([c for c in out.columns if c.endswith("_residual")]) == 1

    def test_missing_col_names_is_an_error(self):
        with pytest.raises(ValueError, match="'col_names' list"):
            run_feature(ProfileSmooth, ramp_profile(), {})

    def test_several_variables_do_not_collide(self):
        out = run_feature(
            ProfileSmooth,
            ramp_profile(),
            {"col_names": ["temp", "psal"], "outputs": ["residual"]},
        )
        assert "temp_residual" in out.columns
        assert "psal_residual" in out.columns


class TestProfileSmoothValues:
    """The numbers the smoother produces."""

    def test_a_clean_ramp_has_no_residual(self):
        """A quadratic fit is exact on a straight line."""
        out = run_feature(
            ProfileSmooth,
            ramp_profile(),
            {"col_names": ["temp"], "outputs": ["residual"]},
        )
        middle = out["temp_residual"].drop_nulls().to_list()
        assert middle == pytest.approx([0.0] * len(middle), abs=1e-9)

    def test_gradient_is_per_decibar_not_per_level(self):
        """One degree per level at 10 dbar spacing is -0.1 per decibar."""
        out = run_feature(
            ProfileSmooth, ramp_profile(), {"col_names": ["temp"], "outputs": ["d1"]}
        )
        values = out["temp_d1"].drop_nulls().to_list()
        assert values == pytest.approx([-0.1] * len(values))

    def test_spacing_column_can_be_switched_off(self):
        out = run_feature(
            ProfileSmooth,
            ramp_profile(),
            {
                "col_names": ["temp"],
                "outputs": ["d1"],
                "params": {"spacing_column": None},
            },
        )
        values = out["temp_d1"].drop_nulls().to_list()
        assert values == pytest.approx([-1.0] * len(values))

    def test_the_spike_is_the_only_outlier(self):
        out = run_feature(
            ProfileSmooth,
            ramp_profile(SPIKED),
            {"col_names": ["temp"], "outputs": ["robust_z"]},
        )
        scores = out["temp_robust_z"].to_list()
        worst = max(
            (index for index, value in enumerate(scores) if value is not None),
            key=lambda index: abs(scores[index]),
        )
        assert worst == 10
        assert abs(scores[10]) > 3.0

    def test_spike_index_reacts_to_the_spike(self):
        out = run_feature(
            ProfileSmooth,
            ramp_profile(SPIKED),
            {"col_names": ["temp"], "outputs": ["spike_index"]},
        )
        values = out["temp_spike_index"].to_list()
        # Level 10 sits 5 above the ramp, and the ramp itself contributes
        # minus half its own gradient: 5 - 1.
        assert values[10] == pytest.approx(4.0)
        assert values[5] == pytest.approx(-1.0)

    def test_outlier_fraction_counts_the_neighbourhood(self):
        """The window around the spike sees it; distant windows do not."""
        out = run_feature(
            ProfileSmooth,
            ramp_profile(SPIKED),
            {
                "col_names": ["temp"],
                "outputs": ["outlier_frac"],
                "params": {"windows": [5]},
            },
        )
        values = out["temp_w5_outlier_frac"].to_list()
        assert values[10] > 0.0
        # Level 15 is far from the spike but still has a defined residual;
        # the first levels have none, so their fraction is null.
        assert values[15] == 0.0
        assert values[0] is None

    def test_curvature_ratio_is_null_where_the_residual_vanishes(self):
        """A perfect fit leaves nothing to take a ratio against."""
        out = run_feature(
            ProfileSmooth,
            ramp_profile(),
            {"col_names": ["temp"], "outputs": ["curvature_ratio"]},
        )
        assert out["temp_curvature_ratio"].drop_nulls().len() == 0

    def test_profile_edges_are_null(self):
        out = run_feature(
            ProfileSmooth,
            ramp_profile(),
            {"col_names": ["temp"], "outputs": ["smooth"], "params": {"window": 5}},
        )
        values = out["temp_smooth"].to_list()
        assert values[:2] == [None, None]
        assert values[-2:] == [None, None]


class TestNeighborDiff:
    """Differences against the levels above and below."""

    def test_column_names_carry_direction_and_lag(self):
        out = run_feature(
            NeighborDiff,
            ramp_profile(),
            {
                "col_names": ["temp"],
                "outputs": ["diff"],
                "params": {"lags": [1, 3], "directions": ["up", "down"]},
            },
        )
        assert "temp_diff_up_1" in out.columns
        assert "temp_diff_down_3" in out.columns
        assert "temp_diff_up_2" not in out.columns

    def test_values_on_a_ramp(self):
        out = run_feature(
            NeighborDiff,
            ramp_profile(),
            {"col_names": ["temp"], "outputs": ["diff"], "params": {"lags": [2]}},
        )
        values = out["temp_diff_up_2"].to_list()
        assert values[:2] == [None, None]
        assert values[2:] == pytest.approx([-2.0] * 19)

    def test_large_difference_fraction_finds_the_spike(self):
        out = run_feature(
            NeighborDiff,
            ramp_profile(NOISY_SPIKED),
            {
                "col_names": ["temp"],
                "outputs": ["large_diff_frac"],
                "params": {"windows": [3]},
            },
        )
        values = out["temp_w3_large_diff_frac"].to_list()
        # The spike shows up as a large difference into it and out of it,
        # so the windows around level 10 are the ones that see anything.
        assert values[10] > 0.0
        assert max(range(21), key=lambda index: values[index] or 0.0) in (9, 10, 11)
        assert all(values[index] == 0.0 for index in range(2, 8))

    def test_robust_scale_is_undefined_on_a_perfectly_regular_profile(self):
        """Identical differences give no scale, so the answer is null.

        Saying "infinitely large" would be a claim the data cannot
        support; an absolute threshold is the way to get an answer here.
        """
        out = run_feature(
            NeighborDiff,
            ramp_profile(SPIKED),
            {
                "col_names": ["temp"],
                "outputs": ["large_diff_frac"],
                "params": {"windows": [3]},
            },
        )
        assert out["temp_w3_large_diff_frac"].drop_nulls().len() == 0

    def test_absolute_threshold_overrides_the_robust_score(self):
        out = run_feature(
            NeighborDiff,
            ramp_profile(),
            {
                "col_names": ["temp"],
                "outputs": ["large_diff_frac"],
                "params": {"windows": [3], "large_diff_threshold": 0.5},
            },
        )
        # Every lag-1 difference on the ramp is 1.0, which clears 0.5.
        assert out["temp_w3_large_diff_frac"].to_list()[10] == pytest.approx(1.0)

    def test_threshold_mapping_must_cover_every_variable(self):
        with pytest.raises(ValueError, match="large_diff_threshold"):
            run_feature(
                NeighborDiff,
                ramp_profile(),
                {
                    "col_names": ["temp", "psal"],
                    "outputs": ["large_diff_frac"],
                    "params": {"large_diff_threshold": {"temp": 0.5}},
                },
            )

    def test_reference_difference_is_not_emitted_unless_asked_for(self):
        out = run_feature(
            NeighborDiff,
            ramp_profile(),
            {
                "col_names": ["temp"],
                "outputs": ["large_diff_frac"],
                "params": {"windows": [3]},
            },
        )
        assert not [c for c in out.columns if c.startswith("temp_diff_")]


class TestRegimeFlags:
    """Where in the water column a level sits."""

    def test_mixed_layer_is_flagged_above_the_step(self):
        """Uniform to level 10, then a sharp change: the top is mixed."""
        values = [10.0] * 10 + [2.0] * 11
        out = run_feature(
            RegimeFlags,
            ramp_profile(values),
            {"outputs": ["in_mixed_layer"], "params": {"reference_pressure": 10.0}},
        )
        flags = out["in_mixed_layer"].to_list()
        assert flags[:10] == [1] * 10
        assert flags[10:] == [0] * 11

    def test_a_uniform_profile_is_mixed_throughout(self):
        out = run_feature(
            RegimeFlags, ramp_profile([10.0] * 21), {"outputs": ["in_mixed_layer"]}
        )
        assert out["in_mixed_layer"].to_list() == [1] * 21

    def test_temperature_criterion_is_available(self):
        values = [10.0] * 10 + [2.0] * 11
        out = run_feature(
            RegimeFlags,
            ramp_profile(values),
            {
                "outputs": ["in_mixed_layer"],
                "params": {"mixed_layer_criterion": "temperature"},
            },
        )
        assert out["in_mixed_layer"].to_list()[:10] == [1] * 10

    def test_unknown_criterion_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown mixed layer criterion"):
            run_feature(
                RegimeFlags,
                ramp_profile(),
                {
                    "outputs": ["in_mixed_layer"],
                    "params": {"mixed_layer_criterion": "salinity"},
                },
            )

    def test_gradient_layer_marks_the_steepest_part(self):
        values = [10.0] * 10 + [2.0] * 11
        out = run_feature(
            RegimeFlags,
            ramp_profile(values),
            {"col_names": ["temp"], "outputs": ["in_gradient_layer"]},
        )
        flags = out["temp_in_gradient_layer"].to_list()
        assert flags[9] == 1 or flags[10] == 1
        assert flags[2] == 0

    def test_normalized_depth_is_zero_at_the_peak(self):
        values = [10.0] * 10 + [2.0] * 11
        out = run_feature(
            RegimeFlags,
            ramp_profile(values),
            {"col_names": ["temp"], "outputs": ["normalized_depth_to_peak_gradient"]},
        )
        position = out["temp_normalized_depth_to_peak_gradient"].to_list()
        peak = min(range(21), key=lambda index: abs(position[index]))
        assert position[peak] == pytest.approx(0.0)
        assert position[0] < 0.0
        assert position[-1] > 0.0

    def test_per_variable_output_needs_col_names(self):
        with pytest.raises(ValueError, match="'col_names' list"):
            run_feature(RegimeFlags, ramp_profile(), {"outputs": ["in_gradient_layer"]})

    def test_mixed_layer_is_not_per_variable(self):
        out = run_feature(
            RegimeFlags,
            ramp_profile(),
            {"col_names": ["temp"], "outputs": ["in_mixed_layer"]},
        )
        assert "in_mixed_layer" in out.columns
        assert "temp_in_mixed_layer" not in out.columns


class TestProfileIsolation:
    """No feature may reach across the boundary between two profiles."""

    @pytest.mark.parametrize(
        "cls, info",
        [
            (ProfileSmooth, {"col_names": ["temp"], "outputs": ["residual"]}),
            (
                NeighborDiff,
                {
                    "col_names": ["temp"],
                    "outputs": ["diff"],
                    "params": {"lags": [1], "directions": ["up"]},
                },
            ),
            (RegimeFlags, {"outputs": ["in_mixed_layer"]}),
        ],
    )
    def test_two_profiles_are_independent(self, cls, info):
        first = ramp_profile()
        second = ramp_profile([5.0] * 21).with_columns(
            pl.lit(2, dtype=pl.Int64).alias("profile_no")
        )
        both = run_feature(cls, first.vstack(second), info)
        alone = run_feature(cls, first, info)
        column = [
            c
            for c in alone.columns
            if c not in ("platform_code", "profile_no", "observation_no")
        ][0]
        assert both.head(21)[column].to_list() == alone[column].to_list()
