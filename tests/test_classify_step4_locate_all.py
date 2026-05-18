"""Unit tests for the classify-side ``LocateDataSetAll`` class.

LocateDataSetAll is the classify-side analogue of the prepare-side
LocateDataSetAll: for each target, it keeps all rows of the input data
(no QC-flag filtering like LocateDataSetA does). Tests cover identity,
output paths, input wiring, the process_targets aggregation, and write
behaviours.

Refactored from a ``unittest.TestCase`` class. Uses ``classify_input_001``
and ``classify_select_001`` fixtures from conftest. Per-target triplication
collapses to ``for tgt in TARGETS_NONEMPTY:`` loops.
"""

import os

import polars as pl
import pytest

from aiqclib.classify.step4_select_rows.dataset_all import LocateDataSetAll

from tests.conftest import TARGETS_NONEMPTY


class TestLocateDataSetAll:
    """Tests for the classify-side LocateDataSetAll's row selection and file output."""

    def test_output_file_names(self, classify_config_001):
        """Default per-target output paths derive from config.path_info."""
        ds = LocateDataSetAll(classify_config_001)
        base = "/path/to/locate_1/nrt_bo_001/locate_folder_1"
        for tgt in TARGETS_NONEMPTY:
            assert (
                str(ds.output_file_names[tgt])
                == f"{base}/selected_rows_classify_{tgt}.parquet"
            )

    def test_step_name(self, classify_config_001):
        """step_name == 'locate'."""
        ds = LocateDataSetAll(classify_config_001)
        assert ds.step_name == "locate"

    def test_input_data_and_selected_profiles(
        self, classify_config_001, classify_input_001, classify_select_001,
    ):
        """input_data and selected_profiles are correctly populated."""
        ds = LocateDataSetAll(
            classify_config_001,
            input_data=classify_input_001.input_data,
            selected_profiles=classify_select_001.selected_profiles,
        )

        assert isinstance(ds.input_data, pl.DataFrame)
        assert ds.input_data.shape[0] == 2456
        assert ds.input_data.shape[1] == 30

        assert isinstance(ds.selected_profiles, pl.DataFrame)
        assert ds.selected_profiles.shape[0] == 10
        assert ds.selected_profiles.shape[1] == 8

    def test_selected_rows(
        self, classify_config_001, classify_input_001, classify_select_001,
    ):
        """process_targets keeps every input row for each target (classify-side: no QC filtering)."""
        ds = LocateDataSetAll(
            classify_config_001,
            input_data=classify_input_001.input_data,
            selected_profiles=classify_select_001.selected_profiles,
        )
        ds.process_targets()

        # No QC filtering on classify side — every target gets all rows.
        for tgt in TARGETS_NONEMPTY:
            assert isinstance(ds.selected_rows[tgt], pl.DataFrame)
            assert ds.selected_rows[tgt].shape[0] == 2456
            assert ds.selected_rows[tgt].shape[1] == 9

    def test_selected_rows_with_empty_input(
        self, classify_config_001, classify_select_001,
    ):
        """process_targets with input_data=None raises ValueError."""
        ds = LocateDataSetAll(
            classify_config_001,
            input_data=None,
            selected_profiles=classify_select_001.selected_profiles,
        )
        with pytest.raises(ValueError):
            ds.process_targets()

    def test_write_selected_rows(
        self, classify_config_001, classify_input_001, classify_select_001,
        test_output_dir,
    ):
        """write_selected_rows produces a parquet per target."""
        ds = LocateDataSetAll(
            classify_config_001,
            input_data=classify_input_001.input_data,
            selected_profiles=classify_select_001.selected_profiles,
        )
        output_paths = {
            tgt: str(test_output_dir / f"test_selected_rows_classify_{tgt}.parquet")
            for tgt in TARGETS_NONEMPTY
        }
        for tgt in TARGETS_NONEMPTY:
            ds.output_file_names[tgt] = output_paths[tgt]

        ds.process_targets()
        ds.write_selected_rows()

        for tgt in TARGETS_NONEMPTY:
            assert os.path.exists(output_paths[tgt])
            os.remove(output_paths[tgt])  # comment out to debug

    def test_write_no_selected_rows(
        self, classify_config_001, classify_input_001, classify_select_001,
    ):
        """write_selected_rows before process_targets raises ValueError."""
        ds = LocateDataSetAll(
            classify_config_001,
            input_data=classify_input_001.input_data,
            selected_profiles=classify_select_001.selected_profiles,
        )
        with pytest.raises(ValueError):
            ds.write_selected_rows()