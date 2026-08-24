"""Unit tests for the ``ConcatDataSetProfile`` class.

ConcatDataSetProfile merges per-profile predictions into one row per profile
(joined on the profile keys, no ``observation_no``), or, with the
``broadcast_to_observations`` step parameter, onto every observation of
each profile.
"""

from datetime import datetime

import polars as pl
import pytest

from aiqclib.classify.step7_concat_datasets.dataset_profile import ConcatDataSetProfile


@pytest.fixture
def profile_input_data():
    """Six observations across three profiles with profile metadata."""
    return pl.DataFrame(
        {
            "platform_code": ["A", "A", "A", "A", "B", "B"],
            "profile_no": [1, 1, 2, 2, 1, 1],
            "observation_no": [1, 2, 1, 2, 1, 2],
            "profile_timestamp": [datetime(2023, 1, d + 1) for d in range(6)],
            "longitude": [10.0, 10.0, 11.0, 11.0, 12.0, 12.0],
            "latitude": [60.0, 60.0, 61.0, 61.0, 62.0, 62.0],
            "temp": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )


@pytest.fixture
def profile_predictions():
    """Per-profile predictions for one target."""
    return {
        "temp": pl.DataFrame(
            {
                "row_id": [1, 2, 3],
                "platform_code": ["A", "A", "B"],
                "profile_no": [1, 2, 1],
                "label": [1, 0, 1],
                "predicted_label": [1, 0, 0],
                "score": [0.9, 0.2, 0.4],
            }
        )
    }


class TestConcatDataSetProfile:
    """Tests for profile-level prediction merging."""

    def test_one_row_per_profile(
        self, classify_config_profile, profile_input_data, profile_predictions
    ):
        ds = ConcatDataSetProfile(
            classify_config_profile,
            input_data=profile_input_data,
            predictions=profile_predictions,
        )
        ds.merge_predictions()

        merged = ds.merged_predictions
        assert merged.height == 3  # one row per profile, not per observation
        assert "observation_no" not in merged.columns
        for col in (
            "platform_code",
            "profile_no",
            "profile_timestamp",
            "longitude",
            "latitude",
            "temp_label",
            "temp_predicted",
            "temp_score",
        ):
            assert col in merged.columns, col

        row = merged.filter(
            (pl.col("platform_code") == "A") & (pl.col("profile_no") == 1)
        )
        assert row["temp_score"][0] == 0.9
        # Profile metadata comes from the input's first row per profile
        assert row["longitude"][0] == 10.0

    def test_broadcast_to_observations(
        self, classify_config_profile, profile_input_data, profile_predictions
    ):
        classify_config_profile.data["step_param_set"]["steps"]["concat"] = {
            "broadcast_to_observations": True
        }
        ds = ConcatDataSetProfile(
            classify_config_profile,
            input_data=profile_input_data,
            predictions=profile_predictions,
        )
        assert ds.broadcast_to_observations is True
        ds.merge_predictions()

        merged = ds.merged_predictions
        assert merged.height == profile_input_data.height
        assert "observation_no" in merged.columns
        # Each observation of profile (A, 1) repeats its profile's prediction
        scores = merged.filter(
            (pl.col("platform_code") == "A") & (pl.col("profile_no") == 1)
        )["temp_score"]
        assert scores.to_list() == [0.9, 0.9]

    def test_default_output_file_name(self, classify_config_profile):
        ds = ConcatDataSetProfile(classify_config_profile)
        assert ds.output_file_name.endswith("predictions_profile.parquet")

    def test_missing_inputs_raise(self, classify_config_profile, profile_predictions):
        ds = ConcatDataSetProfile(
            classify_config_profile, predictions=profile_predictions
        )
        with pytest.raises(ValueError):
            ds.merge_predictions()
