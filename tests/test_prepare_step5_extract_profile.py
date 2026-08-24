"""Unit tests for the ``ExtractDataSetProfile`` class.

ExtractDataSetProfile builds one feature row per profile: profile-native
features (location, day_of_year, profile_summary_stats, profile-level QC
items) join directly, observation-level features are aggregated per profile
via the ``agg`` key, and observation-level features without ``agg`` are
rejected. Uses the select-all pipeline of ``test_dataset_005.yaml`` via the
``dataset_config_profile`` fixtures.
"""

import polars as pl
import pytest

from aiqclib.prepare.step4_select_rows.dataset_profile import LocateDataSetProfile
from aiqclib.prepare.step5_extract_features.dataset_profile import (
    ExtractDataSetProfile,
)

from tests.conftest import TARGETS


def _build_extract(config, dataset_input_005, dataset_select_005, dataset_summary_005):
    ds_locate = LocateDataSetProfile(
        config,
        input_data=dataset_input_005.input_data,
        selected_profiles=dataset_select_005.selected_profiles,
    )
    ds_locate.process_targets()

    ds_extract = ExtractDataSetProfile(
        config,
        input_data=dataset_input_005.input_data,
        selected_profiles=dataset_select_005.selected_profiles,
        selected_rows=ds_locate.selected_rows,
        summary_stats=dataset_summary_005.summary_stats,
        observation_rows=ds_locate.observation_rows,
    )
    return ds_locate, ds_extract


@pytest.fixture
def dataset_summary_005(dataset_config_005, dataset_input_005):
    """step2 summary stats for test_dataset_005.yaml (select-all)."""
    from aiqclib.common.loader.dataset_loader import load_step2_summary_dataset

    ds = load_step2_summary_dataset(dataset_config_005, dataset_input_005.input_data)
    ds.calculate_stats()
    return ds


class TestExtractDataSetProfile:
    """Tests for profile-level feature extraction and aggregation."""

    def test_step_name(self, dataset_config_profile):
        ds = ExtractDataSetProfile(dataset_config_profile)
        assert ds.step_name == "extract"

    def test_one_feature_row_per_profile(
        self,
        dataset_config_profile,
        dataset_input_005,
        dataset_select_005,
        dataset_summary_005,
    ):
        """target_features holds one row per profile with direct and aggregated columns."""
        ds_locate, ds_extract = _build_extract(
            dataset_config_profile,
            dataset_input_005,
            dataset_select_005,
            dataset_summary_005,
        )
        ds_extract.process_targets()

        for tgt in TARGETS:
            features = ds_extract.target_features[tgt]
            assert features.height == ds_locate.selected_rows[tgt].height
            assert "observation_no" not in features.columns
            # Direct profile-level features
            for col in ("longitude", "latitude", "day_of_year", "temp_mean"):
                assert col in features.columns, col
            # Aggregated observation-level features ({col}_{agg})
            for col in ("temp_min", "temp_max", "temp_std", "psal_min", "pres_max"):
                assert col in features.columns, col
            # No nulls in the label
            assert features["label"].null_count() == 0

    def test_missing_agg_is_rejected(
        self,
        dataset_config_profile,
        dataset_input_005,
        dataset_select_005,
        dataset_summary_005,
    ):
        """An observation-level feature without 'agg' raises a naming ValueError."""
        for param in dataset_config_profile.data["feature_param_set"]["params"]:
            if param["feature"] == "basic_values":
                del param["agg"]

        _, ds_extract = _build_extract(
            dataset_config_profile,
            dataset_input_005,
            dataset_select_005,
            dataset_summary_005,
        )
        with pytest.raises(ValueError, match="'basic_values' is observation-level"):
            ds_extract.process_targets()

    def test_unknown_agg_is_rejected(
        self,
        dataset_config_profile,
        dataset_input_005,
        dataset_select_005,
        dataset_summary_005,
    ):
        """An unknown aggregation name raises a naming ValueError."""
        for param in dataset_config_profile.data["feature_param_set"]["params"]:
            if param["feature"] == "basic_values":
                param["agg"] = ["mean", "no_such_agg"]

        _, ds_extract = _build_extract(
            dataset_config_profile,
            dataset_input_005,
            dataset_select_005,
            dataset_summary_005,
        )
        with pytest.raises(ValueError, match="no_such_agg"):
            ds_extract.process_targets()

    def test_observation_qc_item_requires_agg(
        self,
        dataset_config_profile,
        dataset_input_005,
        dataset_select_005,
        dataset_summary_005,
    ):
        """An observation-level QC item without 'agg' is rejected by name."""
        dataset_config_profile.data["feature_param_set"]["params"].append(
            {"feature": "qc_spike", "col_names": ["temp"]}
        )

        _, ds_extract = _build_extract(
            dataset_config_profile,
            dataset_input_005,
            dataset_select_005,
            dataset_summary_005,
        )
        with pytest.raises(ValueError, match="'qc_spike' is observation-level"):
            ds_extract.process_targets()

    def test_qc_items_direct_and_aggregated(
        self,
        dataset_config_profile,
        dataset_input_005,
        dataset_select_005,
        dataset_summary_005,
    ):
        """Profile-level QC items join directly; observation-level ones aggregate."""
        dataset_config_profile.data["feature_param_set"]["params"] += [
            {"feature": "qc_stuck_value", "col_names": ["temp"]},
            {"feature": "qc_spike", "col_names": ["temp"], "agg": ["fail_frac"]},
        ]

        _, ds_extract = _build_extract(
            dataset_config_profile,
            dataset_input_005,
            dataset_select_005,
            dataset_summary_005,
        )
        ds_extract.process_targets()

        for tgt in TARGETS:
            features = ds_extract.target_features[tgt]
            assert "temp_qc_stuck_value" in features.columns
            assert features["temp_qc_stuck_value"].null_count() == 0
            assert "temp_qc_spike_fail_frac" in features.columns
            fail_frac = features["temp_qc_spike_fail_frac"]
            assert fail_frac.dtype == pl.Float64
            assert (fail_frac >= 0.0).all()
            assert (fail_frac <= 1.0).all()

    def test_column_collision_is_rejected(
        self,
        dataset_config_profile,
        dataset_input_005,
        dataset_select_005,
        dataset_summary_005,
    ):
        """A '{col}_{agg}' column colliding with a summary column raises."""
        for param in dataset_config_profile.data["feature_param_set"]["params"]:
            if param["feature"] == "basic_values":
                # profile_summary_stats already produces temp_mean etc.
                param["agg"] = ["mean"]

        _, ds_extract = _build_extract(
            dataset_config_profile,
            dataset_input_005,
            dataset_select_005,
            dataset_summary_005,
        )
        with pytest.raises(ValueError, match="same output column"):
            ds_extract.process_targets()

    def test_missing_observation_rows_raises(
        self,
        dataset_config_profile,
        dataset_input_005,
        dataset_select_005,
        dataset_summary_005,
    ):
        """Aggregation without observation_rows raises a helpful ValueError."""
        ds_locate = LocateDataSetProfile(
            dataset_config_profile,
            input_data=dataset_input_005.input_data,
            selected_profiles=dataset_select_005.selected_profiles,
        )
        ds_locate.process_targets()

        ds_extract = ExtractDataSetProfile(
            dataset_config_profile,
            input_data=dataset_input_005.input_data,
            selected_profiles=dataset_select_005.selected_profiles,
            selected_rows=ds_locate.selected_rows,
            summary_stats=dataset_summary_005.summary_stats,
        )
        with pytest.raises(ValueError, match="Observation rows"):
            ds_extract.process_targets()
