"""Unit tests for the vertical signal processing expressions.

The Savitzky-Golay expectations are the published coefficient sets: the
5-point quadratic smoother is [-3, 12, 17, 12, -3] / 35, its first
derivative is the central difference [-2, -1, 0, 1, 2] / 10, and its second
is [2, -1, -2, -1, 2] / 7. Those are checked directly, and the filters are
then exercised on a profile whose answer is known analytically (a parabola
has a constant second derivative) so that a coefficient applied in the
wrong direction, or across a profile boundary, cannot pass.
"""

import numpy as np
import polars as pl
import pytest

from aiqclib.common.utils import profile_signal as ps


def make_frame(values: dict) -> pl.DataFrame:
    """Build a frame of one or more profiles from {platform: [values]}."""
    rows = {"platform_code": [], "profile_no": [], "observation_no": [], "x": []}
    for platform, series in values.items():
        for index, value in enumerate(series, start=1):
            rows["platform_code"].append(platform)
            rows["profile_no"].append(1)
            rows["observation_no"].append(index)
            rows["x"].append(value)
    return pl.DataFrame(rows)


class TestSavgolCoefficients:
    """The filter weights themselves."""

    def test_quadratic_smoother(self):
        expected = np.array([-3.0, 12.0, 17.0, 12.0, -3.0]) / 35.0
        assert ps.savgol_coefficients(5, 2, 0) == pytest.approx(expected)

    def test_first_derivative_is_the_central_difference(self):
        expected = np.array([-2.0, -1.0, 0.0, 1.0, 2.0]) / 10.0
        assert ps.savgol_coefficients(5, 2, 1) == pytest.approx(expected)

    def test_second_derivative(self):
        expected = np.array([2.0, -1.0, -2.0, -1.0, 2.0]) / 7.0
        assert ps.savgol_coefficients(5, 2, 2) == pytest.approx(expected)

    def test_degree_zero_is_a_moving_average(self):
        assert ps.savgol_coefficients(3, 0, 0) == pytest.approx([1 / 3, 1 / 3, 1 / 3])

    def test_weights_sum_to_one_for_the_smoother(self):
        assert sum(ps.savgol_coefficients(21, 3, 0)) == pytest.approx(1.0)

    @pytest.mark.parametrize(
        "window, polyorder, deriv",
        [(4, 2, 0), (1, 0, 0), (5, 5, 0), (5, 2, 3), (5, 2, -1), (5, -1, 0)],
    )
    def test_invalid_arguments_are_rejected(self, window, polyorder, deriv):
        with pytest.raises(ValueError):
            ps.savgol_coefficients(window, polyorder, deriv)


class TestSavgolExpressions:
    """The filters applied along a profile."""

    def test_smoother_reproduces_a_straight_line(self):
        """A quadratic fit is exact on a line, so smoothing changes nothing."""
        df = make_frame({"a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]})
        out = df.with_columns(ps.savgol_expr("x", 5, 2, 0).alias("s"))
        middle = out["s"].to_list()[2:-2]
        assert middle == pytest.approx([3.0, 4.0, 5.0])

    def test_derivatives_of_a_parabola(self):
        """y = n^2 has first derivative 2n and constant second derivative 2."""
        df = make_frame({"a": [float(n * n) for n in range(1, 8)]})
        out = df.with_columns(
            ps.savgol_expr("x", 5, 2, 1).alias("d1"),
            ps.savgol_expr("x", 5, 2, 2).alias("d2"),
        )
        assert out["d1"].to_list()[2:-2] == pytest.approx([6.0, 8.0, 10.0])
        assert out["d2"].to_list()[2:-2] == pytest.approx([2.0, 2.0, 2.0])

    def test_profile_edges_are_null(self):
        df = make_frame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
        out = df.with_columns(ps.savgol_expr("x", 5, 2, 0).alias("s"))
        assert out["s"].to_list()[0] is None
        assert out["s"].to_list()[-1] is None

    def test_profiles_do_not_bleed_into_each_other(self):
        """The last levels of one profile never feed the first of the next."""
        df = make_frame({"a": [1.0] * 5, "b": [100.0] * 5})
        out = df.with_columns(ps.savgol_expr("x", 3, 1, 0).alias("s"))
        values = out["s"].to_list()
        assert values[1:4] == pytest.approx([1.0, 1.0, 1.0])
        assert values[6:9] == pytest.approx([100.0, 100.0, 100.0])

    def test_unsorted_input_gives_the_same_answer(self):
        """The window orders itself, so the caller cannot get it wrong."""
        df = make_frame({"a": [float(n * n) for n in range(1, 8)]})
        shuffled = df.sample(fraction=1.0, shuffle=True, seed=7)
        out = shuffled.with_columns(ps.savgol_expr("x", 5, 2, 2).alias("d2")).sort(
            "observation_no"
        )
        assert out["d2"].to_list()[2:-2] == pytest.approx([2.0, 2.0, 2.0])

    def test_even_coefficient_count_is_rejected(self):
        with pytest.raises(ValueError, match="odd number of coefficients"):
            ps.fir_expr("x", [0.5, 0.5])


class TestNeighborDifferences:
    """Differences against a neighbouring level."""

    def test_up_and_down_at_lag_one(self):
        df = make_frame({"a": [1.0, 3.0, 6.0, 10.0]})
        out = df.with_columns(
            ps.neighbor_diff_expr("x", 1, "up").alias("up"),
            ps.neighbor_diff_expr("x", 1, "down").alias("down"),
        )
        assert out["up"].to_list() == [None, 2.0, 3.0, 4.0]
        assert out["down"].to_list() == [-2.0, -3.0, -4.0, None]

    def test_larger_lag_loses_more_levels(self):
        df = make_frame({"a": [1.0, 3.0, 6.0, 10.0]})
        out = df.with_columns(ps.neighbor_diff_expr("x", 3, "up").alias("up"))
        assert out["up"].to_list() == [None, None, None, 9.0]

    @pytest.mark.parametrize("lag, direction", [(0, "up"), (1, "sideways")])
    def test_invalid_arguments_are_rejected(self, lag, direction):
        with pytest.raises(ValueError):
            ps.neighbor_diff_expr("x", lag, direction)


class TestCentralDifference:
    """The vertical gradient of one column against another."""

    def test_gradient_of_a_line(self):
        df = make_frame({"a": [0.0, 2.0, 4.0, 6.0]}).with_columns(
            pl.Series("z", [0.0, 1.0, 2.0, 3.0])
        )
        out = df.with_columns(ps.central_difference_expr("x", "z").alias("g"))
        assert out["g"].to_list() == [None, 2.0, 2.0, None]

    def test_zero_separation_gives_null_not_infinity(self):
        """Two levels at the same depth would otherwise divide by zero."""
        df = make_frame({"a": [0.0, 2.0, 4.0, 6.0]}).with_columns(
            pl.Series("z", [1.0, 1.0, 1.0, 3.0])
        )
        out = df.with_columns(
            ps.central_difference_expr("x", "z", min_separation=0.01).alias("g")
        )
        assert out["g"].to_list()[1] is None


class TestSpikeIndex:
    """The Argo spike stencil."""

    def test_isolated_spike_scores_its_own_height(self):
        df = make_frame({"a": [1.0, 1.0, 5.0, 1.0, 1.0]})
        out = df.with_columns(ps.spike_index_column_expr("x").alias("s"))
        assert out["s"].to_list()[2] == pytest.approx(4.0)

    def test_a_pure_gradient_scores_negative(self):
        """The subtracted gradient term is what keeps a ramp off the threshold.

        A straight ramp sits exactly on the average of its neighbours, so
        the first term is zero and the score is minus half the gradient.
        The test flags values *above* a positive threshold, so a steeper
        ramp scores further from failing rather than closer to it.
        """
        df = make_frame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
        out = df.with_columns(ps.spike_index_column_expr("x").alias("s"))
        assert out["s"].to_list()[1:4] == pytest.approx([-1.0, -1.0, -1.0])

    def test_matches_the_rtqc9_item(self):
        """The flag item and the feature share one definition of the value."""
        from aiqclib.prepare.features.qc_spike import QCSpike

        v1, v2, v3 = pl.col("a"), pl.col("b"), pl.col("c")
        df = pl.DataFrame({"a": [1.0], "b": [5.0], "c": [2.0]})
        shared = df.select(ps.spike_index_expr(v1, v2, v3).alias("v"))["v"][0]
        item = df.select(QCSpike().test_value_expr(v1, v2, v3).alias("v"))["v"][0]
        assert shared == item


class TestRollingStatistics:
    """Centred window statistics and the fractions built on them."""

    def test_rolling_median_is_centred(self):
        df = make_frame({"a": [1.0, 2.0, 30.0, 4.0, 5.0]})
        out = df.with_columns(ps.rolling_stat_expr("x", 3, "median").alias("m"))
        assert out["m"].to_list() == [None, 2.0, 4.0, 5.0, None]

    def test_unknown_statistic_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown rolling statistic"):
            ps.rolling_stat_expr("x", 3, "kurtosis")

    def test_even_window_is_rejected(self):
        with pytest.raises(ValueError, match="odd"):
            ps.rolling_stat_expr("x", 4, "mean")

    def test_fraction_counts_the_window(self):
        df = make_frame({"a": [0.0, 0.0, 9.0, 0.0, 0.0]})
        out = df.with_columns(ps.fraction_expr(pl.col("x") > 1.0, 3).alias("f"))
        assert out["f"].to_list() == pytest.approx([0.0, 1 / 3, 1 / 3, 1 / 3, 0.0])


class TestRobustStatistics:
    """MAD and robust z, over a window and over a whole profile."""

    def test_rolling_mad(self):
        df = make_frame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
        out = ps.with_rolling_mad(df, "x", 3, "mad")
        assert out["mad"].to_list() == [None, 1.0, 1.0, 1.0, None]
        assert "_mad_median" not in out.columns

    def test_rolling_mad_is_a_median_of_deviations_from_the_window_median(self):
        """Not a rolling median of each point's own local residual.

        For [1, 2, 30, 4, 5] the window centred on 30 is [2, 30, 4], whose
        median is 4 and whose deviations are [2, 26, 0], so the MAD is 2.
        """
        df = make_frame({"a": [1.0, 2.0, 30.0, 4.0, 5.0]})
        out = ps.with_rolling_mad(df, "x", 3, "mad")
        assert out["mad"].to_list() == [None, 1.0, 2.0, 1.0, None]

    def test_rolling_robust_z_marks_the_outlier(self):
        df = make_frame({"a": [1.0, 2.0, 1.0, 9.0, 1.0, 2.0, 1.0]})
        out = ps.with_rolling_robust_z(df, "x", 5, "z")
        scores = out["z"].to_list()
        assert scores[3] is not None and scores[3] > 3.0

    def test_constant_window_gives_null_not_infinity(self):
        df = make_frame({"a": [2.0, 2.0, 2.0, 2.0, 2.0]})
        out = ps.with_rolling_robust_z(df, "x", 3, "z")
        assert out["z"].to_list()[1:4] == [None, None, None]

    def test_profile_mad_is_constant_within_a_profile(self):
        df = make_frame({"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": [0.0, 0.0, 0.0]})
        out = ps.with_profile_mad(df, "x", "mad")
        assert out.filter(pl.col("platform_code") == "a")["mad"].to_list() == [1.0] * 5
        assert out.filter(pl.col("platform_code") == "b")["mad"].to_list() == [0.0] * 3

    def test_profile_robust_z(self):
        df = make_frame({"a": [1.0, 2.0, 3.0, 4.0, 20.0]})
        out = ps.with_profile_robust_z(df, "x", "z")
        assert out["z"].to_list()[4] == pytest.approx((20.0 - 3.0) / 1.4826)
        assert "_z_mad" not in out.columns

    def test_profile_with_no_spread_gives_null(self):
        """Four identical values give a MAD of zero, which is no scale."""
        df = make_frame({"a": [1.0, 1.0, 1.0, 1.0, 20.0]})
        out = ps.with_profile_robust_z(df, "x", "z")
        assert out["z"].to_list() == [None] * 5
