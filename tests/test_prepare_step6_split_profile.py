"""Unit tests for the ``SplitDataSetProfile`` class.

SplitDataSetProfile splits profile-level feature frames: binary labels fall
through to the label-stratified logic inherited from SplitDataSetAll, while
proportion (float) labels get a plain random test split and uniform k-fold
assignment. Uses the profile pipeline of ``test_dataset_005.yaml`` via the
``dataset_config_profile`` fixtures.
"""

import polars as pl
import pytest

from aiqclib.prepare.step4_select_rows.dataset_profile import LocateDataSetProfile
from aiqclib.prepare.step5_extract_features.dataset_profile import (
    ExtractDataSetProfile,
)
from aiqclib.prepare.step6_split_dataset.dataset_profile import SplitDataSetProfile

from tests.conftest import TARGETS


@pytest.fixture
def profile_target_features(dataset_input_005, dataset_select_005, dataset_config_005):
    """Build profile-level target features for both label modes on demand."""
    from aiqclib.common.loader.dataset_loader import load_step2_summary_dataset

    ds_summary = load_step2_summary_dataset(
        dataset_config_005, dataset_input_005.input_data
    )
    ds_summary.calculate_stats()

    def _build(config):
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
            summary_stats=ds_summary.summary_stats,
            observation_rows=ds_locate.observation_rows,
        )
        ds_extract.process_targets()
        return ds_extract.target_features

    return _build


class TestSplitDataSetProfile:
    """Tests for profile-level train/test splitting and fold assignment."""

    def test_step_name(self, dataset_config_profile):
        ds = SplitDataSetProfile(dataset_config_profile)
        assert ds.step_name == "split"

    def test_binary_split(self, dataset_config_profile, profile_target_features):
        """Binary labels use the inherited stratified split; k_fold in 1..k."""
        ds = SplitDataSetProfile(
            dataset_config_profile,
            target_features=profile_target_features(dataset_config_profile),
        )
        ds.process_targets()

        k_fold = ds.get_k_fold()
        for tgt in TARGETS:
            training_set = ds.training_sets[tgt]
            test_set = ds.test_sets[tgt]
            total = ds.target_features[tgt].height
            assert training_set.height + test_set.height == total
            assert "k_fold" in training_set.columns
            folds = set(training_set["k_fold"].unique().to_list())
            assert folds <= set(range(1, k_fold + 1))
            # Working columns dropped
            for col in ("profile_id", "pair_id"):
                assert col not in training_set.columns
                assert col not in test_set.columns

    def test_proportion_split(
        self, dataset_config_profile_proportion, profile_target_features
    ):
        """Proportion labels get a plain random split and uniform folds."""
        ds = SplitDataSetProfile(
            dataset_config_profile_proportion,
            target_features=profile_target_features(dataset_config_profile_proportion),
        )
        ds.process_targets()

        k_fold = ds.get_k_fold()
        for tgt in TARGETS:
            training_set = ds.training_sets[tgt]
            test_set = ds.test_sets[tgt]
            total = ds.target_features[tgt].height
            assert training_set.height + test_set.height == total
            assert training_set["label"].dtype == pl.Float64
            # No row appears in both sets
            overlap = training_set.join(test_set, on="row_id", how="semi")
            assert overlap.height == 0
            folds = set(training_set["k_fold"].unique().to_list())
            assert folds <= set(range(1, k_fold + 1))
            # k_fold leads the column order, as in the observation pipeline
            assert training_set.columns[0] == "k_fold"
