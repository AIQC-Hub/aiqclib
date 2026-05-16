"""Unit tests for the ``SelectDataSetAll`` class.

SelectDataSetAll is the "select-all" variant of SelectDataSetA — it labels
*every* profile in the input data rather than filtering to positive/negative
pairs. The test config is ``test_dataset_005.yaml``, which uses the
select-all configuration.

Refactored from a single ``unittest.TestCase`` class into a plain pytest
class that uses ``dataset_config_005`` and ``dataset_input_005`` from
conftest. Test outputs go to ``test_output_dir``.
"""

import os

import polars as pl
import pytest

from aiqclib.prepare.step3_select_profiles.dataset_all import SelectDataSetAll


class TestSelectDataSetAll:
    """Tests for SelectDataSetAll's profile-labelling and file output."""

    def test_step_name(self, dataset_config_005):
        """step_name == 'select'."""
        ds = SelectDataSetAll(dataset_config_005)
        assert ds.step_name == "select"

    def test_input_data(self, dataset_config_005, dataset_input_005):
        """input_data is loaded as a Polars DataFrame with the expected shape."""
        ds = SelectDataSetAll(dataset_config_005, input_data=dataset_input_005.input_data)
        assert isinstance(ds.input_data, pl.DataFrame)
        assert ds.input_data.shape[0] == 3267
        assert ds.input_data.shape[1] == 30

    def test_positive_profiles(self, dataset_config_005, dataset_input_005):
        """select_positive_profiles populates pos_profile_df."""
        ds = SelectDataSetAll(dataset_config_005, input_data=dataset_input_005.input_data)
        ds.select_positive_profiles()
        assert isinstance(ds.pos_profile_df, pl.DataFrame)
        assert ds.pos_profile_df.shape[0] == 7
        assert ds.pos_profile_df.shape[1] == 8

    def test_negative_profiles(self, dataset_config_005, dataset_input_005):
        """select_negative_profiles populates neg_profile_df after positives."""
        ds = SelectDataSetAll(dataset_config_005, input_data=dataset_input_005.input_data)
        ds.select_positive_profiles()
        ds.select_negative_profiles()
        assert isinstance(ds.neg_profile_df, pl.DataFrame)
        assert ds.neg_profile_df.shape[0] == 5
        assert ds.neg_profile_df.shape[1] == 8

    def test_label_profiles(self, dataset_config_005, dataset_input_005):
        """label_profiles combines positives + negatives into selected_profiles."""
        ds = SelectDataSetAll(dataset_config_005, input_data=dataset_input_005.input_data)
        ds.label_profiles()
        assert ds.selected_profiles.shape[0] == 12
        assert ds.selected_profiles.shape[1] == 8

    def test_write_selected_profiles(
        self, dataset_config_005, dataset_input_005, test_output_dir
    ):
        """write_selected_profiles produces a parquet at the configured path."""
        ds = SelectDataSetAll(dataset_config_005, input_data=dataset_input_005.input_data)
        output_path = str(test_output_dir / "test_selected_profiles_all.parquet")
        ds.output_file_name = output_path

        ds.label_profiles()
        ds.write_selected_profiles()
        assert os.path.exists(output_path)
        os.remove(output_path)  # comment out to debug

    def test_write_empty_selected_profiles(
        self, dataset_config_005, dataset_input_005, test_output_dir
    ):
        """write_selected_profiles before label_profiles raises ValueError."""
        ds = SelectDataSetAll(dataset_config_005, input_data=dataset_input_005.input_data)
        ds.output_file_name = str(test_output_dir / "test_selected_profiles_all.parquet")

        with pytest.raises(ValueError):
            ds.write_selected_profiles()