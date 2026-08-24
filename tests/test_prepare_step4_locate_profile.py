"""Unit tests for the ``LocateDataSetProfile`` class.

LocateDataSetProfile is the profile-level locate step: per target it keeps
the valid-flagged observation rows (``observation_rows``) and aggregates
them into one labeled row per profile (``selected_rows``), with a binary or
proportion label per the target's ``label_mode``. Uses the select-all
pipeline of ``test_dataset_005.yaml`` via the ``dataset_config_profile``
fixtures.
"""

import os

import polars as pl
import pytest

from aiqclib.prepare.step4_select_rows.dataset_profile import LocateDataSetProfile

from tests.conftest import TARGETS

PROFILE_ROW_COLUMNS = [
    "row_id",
    "profile_id",
    "platform_code",
    "profile_no",
    "label",
    "pair_id",
]

OBSERVATION_ROW_COLUMNS = [
    "row_id",
    "profile_id",
    "platform_code",
    "profile_no",
    "observation_no",
    "pres",
    "flag",
    "label",
    "pair_id",
]


def _build_locate(config, dataset_input_005, dataset_select_005):
    ds = LocateDataSetProfile(
        config,
        input_data=dataset_input_005.input_data,
        selected_profiles=dataset_select_005.selected_profiles,
    )
    ds.process_targets()
    return ds


class TestLocateDataSetProfile:
    """Tests for the profile-level row selection and label aggregation."""

    def test_step_name(self, dataset_config_profile):
        ds = LocateDataSetProfile(dataset_config_profile)
        assert ds.step_name == "locate"

    def test_selected_rows_one_per_profile(
        self, dataset_config_profile, dataset_input_005, dataset_select_005
    ):
        """selected_rows holds one row per valid-flagged profile, no observation_no."""
        ds = _build_locate(
            dataset_config_profile, dataset_input_005, dataset_select_005
        )

        for tgt in TARGETS:
            profiles = ds.selected_rows[tgt]
            assert profiles.columns == PROFILE_ROW_COLUMNS
            assert "observation_no" not in profiles.columns
            # One row per (platform_code, profile_no)
            assert (
                profiles.unique(subset=["platform_code", "profile_no"]).height
                == profiles.height
            )
            # row_id is a unique surrogate
            assert profiles["row_id"].n_unique() == profiles.height

    def test_observation_rows_schema_and_validity(
        self, dataset_config_profile, dataset_input_005, dataset_select_005
    ):
        """observation_rows holds only valid-flagged observations, LocateDataSetAll schema."""
        ds = _build_locate(
            dataset_config_profile, dataset_input_005, dataset_select_005
        )

        for tgt in TARGETS:
            observations = ds.observation_rows[tgt]
            assert observations.columns == OBSERVATION_ROW_COLUMNS
            # Every row carries a resolved 0/1 label (valid flag values only)
            assert observations["label"].null_count() == 0
            assert set(observations["label"].unique().to_list()) <= {0, 1}

    def test_binary_labels(
        self, dataset_config_profile, dataset_input_005, dataset_select_005
    ):
        """Binary label is 1 exactly when a profile has a positive-flagged observation."""
        ds = _build_locate(
            dataset_config_profile, dataset_input_005, dataset_select_005
        )

        for tgt in TARGETS:
            profiles = ds.selected_rows[tgt]
            assert profiles["label"].dtype == pl.UInt32
            expected = (
                ds.observation_rows[tgt]
                .group_by(["platform_code", "profile_no"])
                .agg((pl.col("label").sum() > 0).cast(pl.UInt32).alias("expected"))
            )
            joined = profiles.join(expected, on=["platform_code", "profile_no"])
            assert joined.height == profiles.height
            assert (joined["label"] == joined["expected"]).all()

    def test_proportion_labels(
        self, dataset_config_profile_proportion, dataset_input_005, dataset_select_005
    ):
        """Proportion label is n_pos / n_valid per profile, a float in [0, 1]."""
        ds = _build_locate(
            dataset_config_profile_proportion, dataset_input_005, dataset_select_005
        )

        for tgt in TARGETS:
            profiles = ds.selected_rows[tgt]
            assert profiles["label"].dtype == pl.Float64
            assert (profiles["label"] >= 0.0).all()
            assert (profiles["label"] <= 1.0).all()
            expected = (
                ds.observation_rows[tgt]
                .group_by(["platform_code", "profile_no"])
                .agg(
                    (pl.col("label").sum() / pl.col("label").count()).alias("expected")
                )
            )
            joined = profiles.join(expected, on=["platform_code", "profile_no"])
            assert joined.height == profiles.height
            assert (joined["label"] - joined["expected"]).abs().max() < 1e-12

    def test_binary_and_proportion_agree(
        self,
        dataset_config_profile,
        dataset_config_profile_proportion,
        dataset_input_005,
        dataset_select_005,
    ):
        """A profile has a positive proportion exactly when its binary label is 1."""
        ds_binary = _build_locate(
            dataset_config_profile, dataset_input_005, dataset_select_005
        )
        ds_proportion = _build_locate(
            dataset_config_profile_proportion, dataset_input_005, dataset_select_005
        )

        for tgt in TARGETS:
            joined = ds_binary.selected_rows[tgt].join(
                ds_proportion.selected_rows[tgt],
                on=["platform_code", "profile_no"],
                suffix="_prop",
            )
            assert ((joined["label"] == 1) == (joined["label_prop"] > 0.0)).all()

    def test_write_selected_rows(
        self,
        dataset_config_profile,
        dataset_input_005,
        dataset_select_005,
        test_output_dir,
    ):
        """write_selected_rows produces profile and observation parquets per target."""
        ds = LocateDataSetProfile(
            dataset_config_profile,
            input_data=dataset_input_005.input_data,
            selected_profiles=dataset_select_005.selected_profiles,
        )
        profile_paths = {
            tgt: str(test_output_dir / f"test_selected_rows_profile_{tgt}.parquet")
            for tgt in TARGETS
        }
        observation_paths = {
            tgt: str(test_output_dir / f"test_selected_observation_rows_{tgt}.parquet")
            for tgt in TARGETS
        }
        for tgt in TARGETS:
            ds.output_file_names[tgt] = profile_paths[tgt]
            ds.observation_output_file_names[tgt] = observation_paths[tgt]

        ds.process_targets()
        ds.write_selected_rows()

        for tgt in TARGETS:
            assert os.path.exists(profile_paths[tgt])
            assert os.path.exists(observation_paths[tgt])
            os.remove(profile_paths[tgt])  # comment out to debug
            os.remove(observation_paths[tgt])

    def test_missing_input_raises(self, dataset_config_profile, dataset_select_005):
        """process_targets without input_data raises ValueError."""
        ds = LocateDataSetProfile(
            dataset_config_profile,
            selected_profiles=dataset_select_005.selected_profiles,
        )
        with pytest.raises(ValueError):
            ds.process_targets()

    def test_missing_selected_profiles_raises(
        self, dataset_config_profile, dataset_input_005
    ):
        """process_targets without selected_profiles raises ValueError."""
        ds = LocateDataSetProfile(
            dataset_config_profile,
            input_data=dataset_input_005.input_data,
        )
        with pytest.raises(ValueError):
            ds.process_targets()
