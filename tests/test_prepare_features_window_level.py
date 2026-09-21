"""Unit tests for the window-level feature class and the profile MAD.

The profiles are short arithmetic sequences so every window statistic can
be read off by hand: on [1, 2, 3, 4, 5] a centred window of 3 has median
equal to the level's own value and a median absolute deviation of 1.
"""

import polars as pl
import pytest

from aiqclib.prepare.features.profile_summary import ProfileSummaryStats
from aiqclib.prepare.features.rolling_stats import RollingStats

from tests.test_prepare_features_derived import make_profile, run_feature


class TestRollingStatsOutputs:
    """What the class emits."""

    def test_names_carry_the_window(self):
        out = run_feature(
            RollingStats,
            make_profile([1.0, 2.0, 3.0, 4.0, 5.0]),
            {
                "col_names": ["temp"],
                "outputs": ["median", "mad"],
                "params": {"windows": [3, 5]},
            },
        )
        for window in (3, 5):
            assert f"temp_w{window}_median" in out.columns
            assert f"temp_w{window}_mad" in out.columns

    def test_every_supported_output_by_default(self):
        out = run_feature(
            RollingStats,
            make_profile([float(n) for n in range(1, 8)]),
            {"col_names": ["temp"], "params": {"windows": [3]}},
        )
        for output in ("mean", "median", "mad", "min", "max", "std", "robust_z"):
            assert f"temp_w3_{output}" in out.columns

    def test_variables_do_not_collide(self):
        out = run_feature(
            RollingStats,
            make_profile([1.0, 2.0, 3.0]),
            {
                "col_names": ["temp", "psal"],
                "outputs": ["mean"],
                "params": {"windows": [3]},
            },
        )
        assert "temp_w3_mean" in out.columns
        assert "psal_w3_mean" in out.columns

    def test_missing_col_names_is_an_error(self):
        with pytest.raises(ValueError, match="'col_names' list"):
            run_feature(RollingStats, make_profile([1.0, 2.0, 3.0]), {})

    def test_unknown_output_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown output"):
            run_feature(
                RollingStats,
                make_profile([1.0, 2.0, 3.0]),
                {"col_names": ["temp"], "outputs": ["skew"]},
            )


class TestRollingStatsValues:
    """The numbers themselves."""

    @pytest.fixture
    def ramp(self):
        return make_profile([1.0, 2.0, 3.0, 4.0, 5.0])

    def test_mean_and_median_of_a_ramp(self, ramp):
        out = run_feature(
            RollingStats,
            ramp,
            {
                "col_names": ["temp"],
                "outputs": ["mean", "median"],
                "params": {"windows": [3]},
            },
        )
        assert out["temp_w3_mean"].to_list() == [None, 2.0, 3.0, 4.0, None]
        assert out["temp_w3_median"].to_list() == [None, 2.0, 3.0, 4.0, None]

    def test_mad_of_a_ramp(self, ramp):
        out = run_feature(
            RollingStats,
            ramp,
            {"col_names": ["temp"], "outputs": ["mad"], "params": {"windows": [3]}},
        )
        assert out["temp_w3_mad"].to_list() == [None, 1.0, 1.0, 1.0, None]

    def test_min_and_max(self, ramp):
        out = run_feature(
            RollingStats,
            ramp,
            {
                "col_names": ["temp"],
                "outputs": ["min", "max"],
                "params": {"windows": [3]},
            },
        )
        assert out["temp_w3_min"].to_list() == [None, 1.0, 2.0, 3.0, None]
        assert out["temp_w3_max"].to_list() == [None, 3.0, 4.0, 5.0, None]

    def test_local_robust_z_finds_the_odd_level(self):
        df = make_profile([1.0, 2.0, 30.0, 4.0, 5.0])
        out = run_feature(
            RollingStats,
            df,
            {
                "col_names": ["temp"],
                "outputs": ["robust_z"],
                "params": {"windows": [3]},
            },
        )
        scores = out["temp_w3_robust_z"].to_list()
        assert abs(scores[2]) > 3.0
        assert abs(scores[1]) < abs(scores[2])

    def test_window_wider_than_the_profile_gives_nothing(self):
        out = run_feature(
            RollingStats,
            make_profile([1.0, 2.0, 3.0]),
            {"col_names": ["temp"], "outputs": ["mean"], "params": {"windows": [11]}},
        )
        assert out["temp_w11_mean"].null_count() == 3

    def test_min_samples_relaxes_the_edges(self):
        out = run_feature(
            RollingStats,
            make_profile([1.0, 2.0, 3.0]),
            {
                "col_names": ["temp"],
                "outputs": ["mean"],
                "params": {"windows": [3], "min_samples": 1},
            },
        )
        assert out["temp_w3_mean"].to_list() == [1.5, 2.0, 2.5]

    def test_profiles_are_independent(self):
        first = make_profile([1.0, 2.0, 3.0])
        second = make_profile([100.0, 200.0, 300.0]).with_columns(
            pl.lit(2, dtype=pl.Int64).alias("profile_no")
        )
        out = run_feature(
            RollingStats,
            first.vstack(second),
            {"col_names": ["temp"], "outputs": ["mean"], "params": {"windows": [3]}},
        )
        assert out["temp_w3_mean"].to_list() == [None, 2.0, None, None, 200.0, None]


class TestProfileMad:
    """The profile-level MAD added to profile_summary_stats."""

    @pytest.fixture
    def pieces(self):
        """A target's selected rows and the matching summary table."""
        df = make_profile([1.0, 2.0, 3.0, 4.0, 5.0])
        selected = pl.DataFrame(
            {
                "row_id": [1, 2],
                "platform_code": ["P1", "P1"],
                "profile_no": [1, 1],
                "observation_no": [1, 3],
            }
        )
        summary = pl.DataFrame(
            {
                "platform_code": ["P1"],
                "profile_no": [1],
                "variable": ["temp"],
                "mean": [3.0],
            }
        )
        return df, selected, summary

    def run(self, pieces, stats_names):
        df, selected, summary = pieces
        ds = ProfileSummaryStats(
            target_name="temp",
            feature_info={
                "col_names": ["temp"],
                "summary_stats_names": stats_names,
                "stats_set": {"type": "raw"},
            },
            filtered_input=df,
            selected_rows={"temp": selected},
            summary_stats=summary,
        )
        ds.extract_features()
        return ds.features

    def test_mad_is_computed_from_the_input(self, pieces):
        """[1, 2, 3, 4, 5] has median 3 and deviations [2, 1, 0, 1, 2]."""
        features = self.run(pieces, ["mad"])
        assert features["temp_mad"].to_list() == [1.0, 1.0]

    def test_mad_is_constant_within_the_profile(self, pieces):
        features = self.run(pieces, ["mad"])
        assert features["temp_mad"].n_unique() == 1

    def test_summary_table_statistics_still_work(self, pieces):
        features = self.run(pieces, ["mean", "mad"])
        assert features["temp_mean"].to_list() == [3.0, 3.0]
        assert features["temp_mad"].to_list() == [1.0, 1.0]

    def test_row_count_is_unchanged(self, pieces):
        assert self.run(pieces, ["mad"]).height == 2

    def test_unknown_statistic_is_rejected(self, pieces):
        with pytest.raises(ValueError, match="Unknown summary statistic"):
            self.run(pieces, ["kurtosis"])
