"""Unit tests for the classification-stage ``LocateDataSetProfile`` class.

The classify variant reuses the preparation-stage profile labeling for
targets with a QC flag and adds the label-free path (``skip_evaluation``):
every profile is kept with a null label, and every observation is kept in
``observation_rows`` for feature aggregation.
"""

import polars as pl
import pytest

from aiqclib.classify.step4_select_rows.dataset_profile import LocateDataSetProfile

from tests.conftest import (
    _build_classify_input,
    _build_classify_select,
    TARGETS_NONEMPTY,
)


@pytest.fixture
def classify_profile_chain(classify_config_profile, test_data_file):
    """Input + select outputs for the profile classify config."""
    ds_input = _build_classify_input(classify_config_profile, test_data_file)
    ds_select = _build_classify_select(classify_config_profile, ds_input.input_data)
    return ds_input, ds_select


class TestClassifyLocateDataSetProfile:
    """Tests for the classify-stage profile locate step."""

    def test_step_name_and_file_names(self, classify_config_profile):
        ds = LocateDataSetProfile(classify_config_profile)
        assert ds.step_name == "locate"
        for tgt in TARGETS_NONEMPTY:
            assert ds.output_file_names[tgt].endswith(
                f"selected_rows_classify_{tgt}.parquet"
            )
            assert ds.observation_output_file_names[tgt].endswith(
                f"selected_observation_rows_classify_{tgt}.parquet"
            )

    def test_labelled_targets(self, classify_config_profile, classify_profile_chain):
        """Targets with a flag get the preparation-style profile labels."""
        ds_input, ds_select = classify_profile_chain
        ds = LocateDataSetProfile(
            classify_config_profile,
            input_data=ds_input.input_data,
            selected_profiles=ds_select.selected_profiles,
        )
        ds.process_targets()

        for tgt in TARGETS_NONEMPTY:
            profiles = ds.selected_rows[tgt]
            assert "observation_no" not in profiles.columns
            assert profiles["label"].null_count() == 0
            assert set(profiles["label"].unique().to_list()) <= {0, 1}
            # Only valid-flagged observations kept for aggregation
            assert ds.observation_rows[tgt]["label"].null_count() == 0

    def test_label_free_target(self, classify_config_profile, classify_profile_chain):
        """A target without a flag keeps every profile with a null label."""
        ds_input, ds_select = classify_profile_chain
        for variable in classify_config_profile.data["target_set"]["variables"]:
            if variable["name"] == "psal":
                variable["flag"] = None

        ds = LocateDataSetProfile(
            classify_config_profile,
            input_data=ds_input.input_data,
            selected_profiles=ds_select.selected_profiles,
        )
        ds.process_targets()

        all_profiles = (
            ds_input.input_data.select(["platform_code", "profile_no"]).unique().height
        )
        profiles = ds.selected_rows["psal"]
        assert profiles.height == all_profiles
        assert profiles["label"].null_count() == profiles.height
        # Every observation kept (label-free aggregation input)
        observations = ds.observation_rows["psal"]
        assert observations.height == ds_input.input_data.height
        assert observations["label"].null_count() == observations.height

        # The labelled target is unaffected
        assert ds.selected_rows["temp"]["label"].null_count() == 0

    def test_proportion_labels(
        self, classify_config_profile_proportion, test_data_file
    ):
        """label_mode: proportion produces float labels at classify time too."""
        ds_input = _build_classify_input(
            classify_config_profile_proportion, test_data_file
        )
        ds_select = _build_classify_select(
            classify_config_profile_proportion, ds_input.input_data
        )
        ds = LocateDataSetProfile(
            classify_config_profile_proportion,
            input_data=ds_input.input_data,
            selected_profiles=ds_select.selected_profiles,
        )
        ds.process_targets()

        for tgt in TARGETS_NONEMPTY:
            labels = ds.selected_rows[tgt]["label"]
            assert labels.dtype == pl.Float64
            assert (labels >= 0.0).all()
            assert (labels <= 1.0).all()
