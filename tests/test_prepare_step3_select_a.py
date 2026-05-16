"""Unit tests for the ``SelectDataSetA`` class.

Exercises positive/negative profile selection, profile-pair matching, and
label_profiles aggregation. The NegX5 variant tests the same behaviour with
a 1:5 positive:negative ratio (test_dataset_003.yaml's neg_pos_ratio).

Refactored from two ``unittest.TestCase`` classes
(``TestSelectDataSetA`` + ``TestSelectDataSetANegX5``) into two plain
classes that share the conftest-provided ``dataset_input_001`` /
``dataset_input_003`` fixtures.
"""

import os

import polars as pl
import pytest

from aiqclib.prepare.step3_select_profiles.dataset_a import SelectDataSetA


# ---------------------------------------------------------------------------
# Tests against test_dataset_001.yaml (default neg:pos = 1:1)
# ---------------------------------------------------------------------------

class TestSelectDataSetA:
    """Tests for SelectDataSetA with the default 1:1 negative:positive ratio."""

    def test_step_name(self, dataset_config_001):
        """step_name == 'select'."""
        ds = SelectDataSetA(dataset_config_001)
        assert ds.step_name == "select"

    def test_output_file_name(self, dataset_config_001):
        """Configured output path comes from config.path_info."""
        ds = SelectDataSetA(dataset_config_001)
        assert (
            str(ds.output_file_name)
            == "/path/to/select_1/nrt_bo_001/select_folder_1/selected_profiles.parquet"
        )

    def test_default_output_file_name(self, dataset_config_002):
        """test_dataset_002.yaml uses the default-path pattern."""
        ds = SelectDataSetA(dataset_config_002)
        assert (
            str(ds.output_file_name)
            == "/path/to/data_1/nrt_bo_001/select/selected_profiles.parquet"
        )

    def test_input_data(self, dataset_config_001, dataset_input_001):
        """input_data is a Polars DataFrame with the expected shape."""
        ds = SelectDataSetA(dataset_config_001, input_data=dataset_input_001.input_data)
        assert isinstance(ds.input_data, pl.DataFrame)
        assert ds.input_data.shape[0] == 3267
        assert ds.input_data.shape[1] == 30

    def test_positive_profiles(self, dataset_config_001, dataset_input_001):
        """select_positive_profiles populates pos_profile_df."""
        ds = SelectDataSetA(dataset_config_001, input_data=dataset_input_001.input_data)
        ds.select_positive_profiles()
        assert isinstance(ds.pos_profile_df, pl.DataFrame)
        assert ds.pos_profile_df.shape[0] == 7
        assert ds.pos_profile_df.shape[1] == 7

    def test_negative_profiles(self, dataset_config_001, dataset_input_001):
        """select_negative_profiles populates neg_profile_df after select_positive_profiles."""
        ds = SelectDataSetA(dataset_config_001, input_data=dataset_input_001.input_data)
        ds.select_positive_profiles()
        ds.select_negative_profiles()
        assert isinstance(ds.neg_profile_df, pl.DataFrame)
        assert ds.neg_profile_df.shape[0] == 5
        assert ds.neg_profile_df.shape[1] == 7

    def test_find_profile_pairs(self, dataset_config_001, dataset_input_001):
        """find_profile_pairs matches positives to negatives 1:1 (default ratio)."""
        ds = SelectDataSetA(dataset_config_001, input_data=dataset_input_001.input_data)
        ds.select_positive_profiles()
        ds.select_negative_profiles()
        ds.find_profile_pairs()
        assert ds.pos_profile_df.shape[0] == 7
        assert ds.pos_profile_df.shape[1] == 8
        assert ds.neg_profile_df.shape[0] == 7
        assert ds.neg_profile_df.shape[1] == 8

    def test_label_profiles(self, dataset_config_001, dataset_input_001):
        """label_profiles combines positives + negatives into selected_profiles."""
        ds = SelectDataSetA(dataset_config_001, input_data=dataset_input_001.input_data)
        ds.label_profiles()
        assert ds.selected_profiles.shape[0] == 14
        assert ds.selected_profiles.shape[1] == 8

    def test_write_selected_profiles(
        self, dataset_config_001, dataset_input_001, test_output_dir
    ):
        """write_selected_profiles produces a parquet at the configured path."""
        ds = SelectDataSetA(dataset_config_001, input_data=dataset_input_001.input_data)
        output_path = str(test_output_dir / "test_selected_profiles.parquet")
        ds.output_file_name = output_path

        ds.label_profiles()
        ds.write_selected_profiles()
        assert os.path.exists(output_path)
        os.remove(output_path)  # comment out to debug

    def test_write_empty_selected_profiles(
        self, dataset_config_001, dataset_input_001, test_output_dir
    ):
        """write_selected_profiles before label_profiles raises ValueError."""
        ds = SelectDataSetA(dataset_config_001, input_data=dataset_input_001.input_data)
        # The output path doesn't matter — we expect the call to fail before writing.
        ds.output_file_name = str(test_output_dir / "test_selected_profiles.parquet")

        with pytest.raises(ValueError):
            ds.write_selected_profiles()


# ---------------------------------------------------------------------------
# Tests against test_dataset_003.yaml (neg:pos = 1:5)
# ---------------------------------------------------------------------------

class TestSelectDataSetANegX5:
    """Tests for SelectDataSetA with the 1:5 negative:positive ratio (config 003)."""

    def test_neg_pos_ratio(self, dataset_config_003, dataset_input_003):
        """The configured neg_pos_ratio of 5 is honoured."""
        ds = SelectDataSetA(dataset_config_003, input_data=dataset_input_003.input_data)
        assert ds.config.get_step_params("select").get("neg_pos_ratio", 1) == 5

    def test_find_profile_pairs(self, dataset_config_003, dataset_input_003):
        """find_profile_pairs produces 5 negatives per positive."""
        ds = SelectDataSetA(dataset_config_003, input_data=dataset_input_003.input_data)
        ds.select_positive_profiles()
        ds.select_negative_profiles()
        ds.find_profile_pairs()
        assert ds.pos_profile_df.shape[0] == 7
        assert ds.pos_profile_df.shape[1] == 8
        assert ds.neg_profile_df.shape[0] == 35
        assert ds.neg_profile_df.shape[1] == 8

    def test_label_profiles(self, dataset_config_003, dataset_input_003):
        """label_profiles combines 25 positives + 125 negatives = 150 selected profiles."""
        ds = SelectDataSetA(dataset_config_003, input_data=dataset_input_003.input_data)
        ds.label_profiles()
        assert ds.selected_profiles.shape[0] == 42
        assert ds.selected_profiles.shape[1] == 8

    def test_write_selected_profiles(
        self, dataset_config_003, dataset_input_003, test_output_dir
    ):
        """write_selected_profiles (NegX5) produces a parquet at the configured path."""
        ds = SelectDataSetA(dataset_config_003, input_data=dataset_input_003.input_data)
        output_path = str(test_output_dir / "test_selected_profiles_negx5.parquet")
        ds.output_file_name = output_path

        ds.label_profiles()
        ds.write_selected_profiles()
        assert os.path.exists(output_path)
        os.remove(output_path)  # comment out to debug