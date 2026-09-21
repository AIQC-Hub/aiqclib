"""End-to-end checks that the new feature groups reach the model frame.

The unit tests for ``derived_values`` and ``stratification`` run the
classes on synthetic profiles. These run them through the real prepare
pipeline on the fixture dataset instead, which is what catches the wiring
mistakes a unit test cannot see: a feature that is registered but not
loadable, columns that collide with another feature's, a join that loses
rows, or a class that works observation-level but not once aggregated per
profile.
"""

import pytest

from tests.conftest import _load_dataset_config, build_prepare_pipeline
from tests.test_prepare_step5_extract_profile import _build_extract

SIGNAL_FEATURES = [
    {
        "feature": "derived_values",
        "outputs": ["sigma0", "depth"],
        "stats_set": {"type": "raw"},
    },
    {
        "feature": "stratification",
        "outputs": ["n2", "n2_abs", "unstable_flag"],
        "stats_set": {"type": "raw"},
    },
    {
        "feature": "profile_smooth",
        "col_names": ["temp", "psal"],
        "outputs": ["residual", "robust_z", "spike_index", "outlier_frac"],
        "params": {"window": 5, "windows": [5]},
        "stats_set": {"type": "raw"},
    },
    {
        "feature": "neighbor_diff",
        "col_names": ["temp", "psal"],
        "outputs": ["diff"],
        "params": {"lags": [1, 2], "directions": ["up", "down"]},
        "stats_set": {"type": "raw"},
    },
    {
        "feature": "regime_flags",
        "col_names": ["temp"],
        "stats_set": {"type": "raw"},
    },
    {
        "feature": "rolling_stats",
        "col_names": ["temp"],
        "outputs": ["median", "mad", "robust_z"],
        "params": {"windows": [5]},
        "stats_set": {"type": "raw"},
    },
]

NEW_COLUMNS = [
    "sigma0",
    "depth",
    "n2",
    "n2_abs",
    "unstable_flag",
    "temp_residual",
    "temp_robust_z",
    "temp_spike_index",
    "temp_w5_outlier_frac",
    "temp_diff_up_1",
    "psal_diff_down_2",
    "in_mixed_layer",
    "temp_in_gradient_layer",
    "temp_normalized_depth_to_peak_gradient",
    "temp_w5_median",
    "temp_w5_mad",
    "temp_w5_robust_z",
]


def with_signal_features(config, agg=None):
    """Append the new feature groups to a dataset configuration."""
    for entry in SIGNAL_FEATURES:
        param = dict(entry)
        if agg is not None:
            param["agg"] = list(agg)
        config.data["feature_param_set"]["params"].append(param)
    return config


class TestObservationLevelPipeline:
    """The features as ordinary observation-level columns."""

    @pytest.fixture
    def features(self, dataset_config_001, test_data_file):
        config = with_signal_features(dataset_config_001)
        return build_prepare_pipeline(config, test_data_file).extract.target_features

    def test_new_columns_reach_every_target(self, features):
        assert features
        for target_name, frame in features.items():
            for column in NEW_COLUMNS:
                assert column in frame.columns, f"{column} missing for {target_name}"

    def test_no_rows_are_lost(self, test_data_file):
        """Adding a feature must not drop rows through its join."""
        plain = build_prepare_pipeline(
            _load_dataset_config("test_dataset_001.yaml"), test_data_file
        )
        enriched = build_prepare_pipeline(
            with_signal_features(_load_dataset_config("test_dataset_001.yaml")),
            test_data_file,
        )
        for target_name, frame in enriched.extract.target_features.items():
            assert frame.height == plain.extract.target_features[target_name].height

    def test_density_is_oceanographic(self, features):
        """Sigma-0 in the fixture region sits in the usual range for sea water."""
        frame = next(iter(features.values()))
        values = frame["sigma0"].drop_nulls()
        assert values.len() > 0
        assert 0.0 < values.min() and values.max() < 40.0

    def test_stability_flag_is_a_flag(self, features):
        frame = next(iter(features.values()))
        assert set(frame["unstable_flag"].drop_nulls().unique().to_list()) <= {0, 1}

    def test_magnitude_matches_the_value(self, features):
        frame = next(iter(features.values())).drop_nulls(["n2", "n2_abs"])
        assert frame["n2_abs"].to_list() == pytest.approx(frame["n2"].abs().to_list())

    def test_depth_is_never_null_where_pressure_is_not(self, features):
        """Depth needs only pressure and latitude, which every row has."""
        frame = next(iter(features.values()))
        assert frame["depth"].null_count() == 0


@pytest.fixture
def dataset_summary_005(dataset_config_005, dataset_input_005):
    """step2 summary stats for the select-all configuration."""
    from aiqclib.common.loader.dataset_loader import load_step2_summary_dataset

    ds = load_step2_summary_dataset(dataset_config_005, dataset_input_005.input_data)
    ds.calculate_stats()
    return ds


class TestProfileLevelPipeline:
    """The same features aggregated per profile."""

    @pytest.fixture
    def extract(
        self,
        dataset_config_profile,
        dataset_input_005,
        dataset_select_005,
        dataset_summary_005,
    ):
        """The profile-level extract step with the new features added."""

        def build(agg=None):
            config = with_signal_features(dataset_config_profile, agg=agg)
            _, ds_extract = _build_extract(
                config, dataset_input_005, dataset_select_005, dataset_summary_005
            )
            return ds_extract

        return build

    def test_aggregated_columns_are_named_after_the_aggregation(self, extract):
        ds = extract(agg=["mean", "max"])
        ds.process_targets()
        frame = next(iter(ds.target_features.values()))
        for column in NEW_COLUMNS:
            assert f"{column}_mean" in frame.columns
            assert f"{column}_max" in frame.columns
            assert column not in frame.columns

    def test_one_row_per_profile(self, extract):
        ds = extract(agg=["mean"])
        ds.process_targets()
        frame = next(iter(ds.target_features.values()))
        assert frame.select(["platform_code", "profile_no"]).n_unique() == frame.height

    def test_aggregated_density_is_still_oceanographic(self, extract):
        """Aggregation must not turn nulls at profile edges into zeros."""
        ds = extract(agg=["mean"])
        ds.process_targets()
        values = next(iter(ds.target_features.values()))["sigma0_mean"].drop_nulls()
        assert values.len() > 0
        assert 0.0 < values.min() and values.max() < 40.0

    def test_unaggregated_use_is_an_error(self, extract):
        """Without agg the mismatch is reported, not silently worked around."""
        ds = extract(agg=None)
        with pytest.raises(ValueError, match="observation-level"):
            ds.process_targets()
