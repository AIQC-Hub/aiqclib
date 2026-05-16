"""Unit tests for the ``LocateDataSetAll`` class.

LocateDataSetAll is the "select-all" variant of LocateDataSetA — for each
target, it keeps all rows of the (select-all) input rather than filtering
by QC flag values. Used by tests under ``test_dataset_005.yaml``.

Refactored from a single ``unittest.TestCase`` class. The three target_value
dicts (one per target) are replaced by the same module-level
``TARGET_VALUES`` constant used in ``test_prepare_step4_locate_a.py``, and
per-target triplication collapses to ``for tgt in TARGETS:`` loops.
"""

import os

import polars as pl
import pytest

from aiqclib.prepare.step4_select_rows.dataset_all import LocateDataSetAll

from tests.conftest import TARGETS


# Module-level target-value specifications: same shape, different QC column
# per target. Identical to test_prepare_step4_locate_a.py's TARGET_VALUES.
TARGET_VALUES: dict[str, dict] = {
    tgt: {
        "flag": f"{tgt}_qc",
        "pos_flag_values": [4],
        "neg_flag_values": [1],
    }
    for tgt in TARGETS
}


class TestLocateDataSetAll:
    """Tests for LocateDataSetAll's row selection and file output (select-all variant)."""

    def test_step_name(self, dataset_config_005):
        """step_name == 'locate'."""
        ds = LocateDataSetAll(dataset_config_005)
        assert ds.step_name == "locate"

    def test_input_data_and_selected_profiles(
        self, dataset_config_005, dataset_input_005, dataset_select_005
    ):
        """input_data and selected_profiles are correctly populated."""
        ds = LocateDataSetAll(
            dataset_config_005,
            input_data=dataset_input_005.input_data,
            selected_profiles=dataset_select_005.selected_profiles,
        )

        assert isinstance(ds.input_data, pl.DataFrame)
        assert ds.input_data.shape[0] == 3267
        assert ds.input_data.shape[1] == 30

        assert isinstance(ds.selected_profiles, pl.DataFrame)
        assert ds.selected_profiles.shape[0] == 12
        assert ds.selected_profiles.shape[1] == 8

    def test_selected_rows(
        self, dataset_config_005, dataset_input_005, dataset_select_005
    ):
        """process_targets keeps every input row for each target (select-all)."""
        ds = LocateDataSetAll(
            dataset_config_005,
            input_data=dataset_input_005.input_data,
            selected_profiles=dataset_select_005.selected_profiles,
        )
        ds.process_targets()

        for tgt in TARGETS:
            assert isinstance(ds.selected_rows[tgt], pl.DataFrame)
            assert ds.selected_rows[tgt].shape[0] == 3267
            assert ds.selected_rows[tgt].shape[1] == 9

    def test_write_selected_rows(
        self, dataset_config_005, dataset_input_005, dataset_select_005, test_output_dir
    ):
        """write_selected_rows produces a parquet per target."""
        ds = LocateDataSetAll(
            dataset_config_005,
            input_data=dataset_input_005.input_data,
            selected_profiles=dataset_select_005.selected_profiles,
        )
        output_paths = {
            tgt: str(test_output_dir / f"test_selected_rows_all_{tgt}.parquet")
            for tgt in TARGETS
        }
        for tgt in TARGETS:
            ds.output_file_names[tgt] = output_paths[tgt]

        ds.process_targets()
        ds.write_selected_rows()

        for tgt in TARGETS:
            assert os.path.exists(output_paths[tgt])
            os.remove(output_paths[tgt])  # comment out to debug

    def test_write_no_selected_rows(
        self, dataset_config_005, dataset_input_005, dataset_select_005
    ):
        """write_selected_rows before process_targets raises ValueError."""
        ds = LocateDataSetAll(
            dataset_config_005,
            input_data=dataset_input_005.input_data,
            selected_profiles=dataset_select_005.selected_profiles,
        )
        with pytest.raises(ValueError):
            ds.write_selected_rows()