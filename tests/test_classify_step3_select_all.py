"""Unit tests for the ``SelectDataSetAll`` class.

SelectDataSetAll is the classify-side analogue of SelectDataSetAll on the
prepare side: it selects every profile in the input data (no positive /
negative pair logic). Both ``select_all_profiles()`` and ``label_profiles()``
populate ``selected_profiles``; tests verify both entry points.

Refactored from a ``unittest.TestCase`` class. Renamed
``TestSelectDataSetA`` → ``TestSelectDataSetAll`` (the original was a
copy-paste from another test file — it tests SelectDataSetAll, not
SelectDataSetA). ``pytest -k SelectDataSetAll`` filters will now match.
"""

import os

import polars as pl
import pytest

from aiqclib.classify.step3_select_profiles.dataset_all import SelectDataSetAll


class TestSelectDataSetAll:
    """Tests for SelectDataSetAll's profile-selection and file output.

    Renamed from ``TestSelectDataSetA`` — the original was a copy-paste
    from the prepare-side equivalent. This file tests classify-side
    SelectDataSetAll.
    """

    def test_step_name(self, classify_config_001):
        """step_name == 'select'."""
        ds = SelectDataSetAll(classify_config_001)
        assert ds.step_name == "select"

    def test_output_file_name(self, classify_config_001):
        """Configured output path comes from config.path_info."""
        ds = SelectDataSetAll(classify_config_001)
        assert (
            str(ds.output_file_name)
            == "/path/to/select_1/nrt_bo_001/select_folder_1/selected_profiles_classify.parquet"
        )

    def test_input_data(self, classify_config_001, classify_input_001):
        """input_data is a Polars DataFrame with the expected shape."""
        ds = SelectDataSetAll(classify_config_001, input_data=classify_input_001.input_data)
        assert isinstance(ds.input_data, pl.DataFrame)
        assert ds.input_data.shape[0] == 2456
        assert ds.input_data.shape[1] == 30

    def test_selected_profiles(self, classify_config_001, classify_input_001):
        """select_all_profiles populates selected_profiles with every profile."""
        ds = SelectDataSetAll(classify_config_001, input_data=classify_input_001.input_data)
        ds.select_all_profiles()
        assert isinstance(ds.selected_profiles, pl.DataFrame)
        assert ds.selected_profiles.shape[0] == 10
        assert ds.selected_profiles.shape[1] == 8

    def test_label_profiles(self, classify_config_001, classify_input_001):
        """label_profiles produces the same shape as select_all_profiles."""
        ds = SelectDataSetAll(classify_config_001, input_data=classify_input_001.input_data)
        ds.label_profiles()
        assert ds.selected_profiles.shape[0] == 10
        assert ds.selected_profiles.shape[1] == 8

    def test_write_selected_profiles(
        self, classify_config_001, classify_input_001, test_output_dir,
    ):
        """write_selected_profiles produces a parquet at the configured path."""
        ds = SelectDataSetAll(classify_config_001, input_data=classify_input_001.input_data)
        output_path = str(test_output_dir / "test_selected_profiles_classify.parquet")
        ds.output_file_name = output_path

        ds.label_profiles()
        ds.write_selected_profiles()
        assert os.path.exists(output_path)
        os.remove(output_path)  # comment out to debug

    def test_write_empty_selected_profiles(
        self, classify_config_001, classify_input_001, test_output_dir,
    ):
        """write_selected_profiles before label_profiles raises ValueError."""
        ds = SelectDataSetAll(classify_config_001, input_data=classify_input_001.input_data)
        ds.output_file_name = str(test_output_dir / "test_selected_profiles_classify.parquet")

        with pytest.raises(ValueError):
            ds.write_selected_profiles()