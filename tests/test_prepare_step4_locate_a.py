"""Unit tests for the ``LocateDataSetA`` class.

LocateDataSetA selects rows within each profile that match per-target QC
flag values (e.g. ``temp_qc == 4`` for positive rows, ``temp_qc == 1`` for
negatives). Tests cover positive/negative row selection per target, the
combined ``selected_rows`` output, and the write/empty-state error paths.

The NegX5 variant uses test_dataset_003.yaml, which has a
``neg_x_multiplier`` setting that scales up the number of negative rows
selected.

Refactored from two ``unittest.TestCase`` classes into two plain classes
that share the conftest-provided ``dataset_input_001`` /
``dataset_input_003`` and ``dataset_select_001`` / ``dataset_select_003``
fixtures. Per-target triplication (temp/psal/pres assertions written three
times each) collapses to ``for tgt in TARGETS:`` loops.
"""

import os

import polars as pl
import pytest

from aiqclib.prepare.step4_select_rows.dataset_a import LocateDataSetA

from tests.conftest import TARGETS


# Module-level target-value specifications: same shape, different QC column
# per target. Previously hand-coded as three separate dict attributes on
# each test class.
TARGET_VALUES: dict[str, dict] = {
    tgt: {
        "flag": f"{tgt}_qc",
        "pos_flag_values": [4],
        "neg_flag_values": [1],
    }
    for tgt in TARGETS
}


# ---------------------------------------------------------------------------
# Tests against test_dataset_001.yaml (default — no neg_x_multiplier)
# ---------------------------------------------------------------------------

class TestLocateDataSetA:
    """Tests for LocateDataSetA with the default config."""

    def test_output_file_names(self, dataset_config_001):
        """Configured output paths come from config.path_info, per-target."""
        ds = LocateDataSetA(dataset_config_001)
        base = "/path/to/locate_1/nrt_bo_001/locate_folder_1"
        # Original asserted only temp and psal; extending to all three.
        for tgt in TARGETS:
            assert (
                str(ds.output_file_names[tgt])
                == f"{base}/selected_rows_{tgt}.parquet"
            )

    def test_step_name(self, dataset_config_001):
        """step_name == 'locate'."""
        ds = LocateDataSetA(dataset_config_001)
        assert ds.step_name == "locate"

    def test_input_data_and_selected_profiles(
        self, dataset_config_001, dataset_input_001, dataset_select_001
    ):
        """input_data and selected_profiles are correctly populated."""
        ds = LocateDataSetA(
            dataset_config_001,
            input_data=dataset_input_001.input_data,
            selected_profiles=dataset_select_001.selected_profiles,
        )

        assert isinstance(ds.input_data, pl.DataFrame)
        assert ds.input_data.shape[0] == 3267
        assert ds.input_data.shape[1] == 30

        assert isinstance(ds.selected_profiles, pl.DataFrame)
        assert ds.selected_profiles.shape[0] == 14
        assert ds.selected_profiles.shape[1] == 8

    def test_positive_rows(
        self, dataset_config_001, dataset_input_001, dataset_select_001
    ):
        """select_positive_rows populates positive_rows per target."""
        ds = LocateDataSetA(
            dataset_config_001,
            input_data=dataset_input_001.input_data,
            selected_profiles=dataset_select_001.selected_profiles,
        )
        for tgt in TARGETS:
            ds.select_positive_rows(tgt, TARGET_VALUES[tgt])

        expected_pos = {"temp": 12, "psal": 18, "pres": 9}
        for tgt in TARGETS:
            assert isinstance(ds.positive_rows[tgt], pl.DataFrame)
            assert ds.positive_rows[tgt].shape[0] == expected_pos[tgt]
            assert ds.positive_rows[tgt].shape[1] == 9

    def test_negative_rows(
        self, dataset_config_001, dataset_input_001, dataset_select_001
    ):
        """select_negative_rows populates negative_rows per target (after positives)."""
        ds = LocateDataSetA(
            dataset_config_001,
            input_data=dataset_input_001.input_data,
            selected_profiles=dataset_select_001.selected_profiles,
        )
        for tgt in TARGETS:
            ds.select_positive_rows(tgt, TARGET_VALUES[tgt])
            ds.select_negative_rows(tgt, TARGET_VALUES[tgt])

        expected_neg = {"temp": 12, "psal": 18, "pres": 9}
        for tgt in TARGETS:
            assert isinstance(ds.negative_rows[tgt], pl.DataFrame)
            assert ds.negative_rows[tgt].shape[0] == expected_neg[tgt]
            assert ds.negative_rows[tgt].shape[1] == 8

    def test_selected_rows(
        self, dataset_config_001, dataset_input_001, dataset_select_001
    ):
        """process_targets aggregates positive+negative rows per target."""
        ds = LocateDataSetA(
            dataset_config_001,
            input_data=dataset_input_001.input_data,
            selected_profiles=dataset_select_001.selected_profiles,
        )
        ds.process_targets()

        expected = {"temp": 24, "psal": 36, "pres": 18}
        for tgt in TARGETS:
            assert isinstance(ds.selected_rows[tgt], pl.DataFrame)
            assert ds.selected_rows[tgt].shape[0] == expected[tgt]
            assert ds.selected_rows[tgt].shape[1] == 9

    def test_write_selected_rows(
        self, dataset_config_001, dataset_input_001, dataset_select_001, test_output_dir
    ):
        """write_selected_rows produces a parquet per target."""
        ds = LocateDataSetA(
            dataset_config_001,
            input_data=dataset_input_001.input_data,
            selected_profiles=dataset_select_001.selected_profiles,
        )
        output_paths = {
            tgt: str(test_output_dir / f"test_selected_rows_{tgt}.parquet")
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
        self, dataset_config_001, dataset_input_001, dataset_select_001
    ):
        """write_selected_rows before process_targets raises ValueError."""
        ds = LocateDataSetA(
            dataset_config_001,
            input_data=dataset_input_001.input_data,
            selected_profiles=dataset_select_001.selected_profiles,
        )
        with pytest.raises(ValueError):
            ds.write_selected_rows()


# ---------------------------------------------------------------------------
# Tests against test_dataset_003.yaml (NegX5 — neg_x_multiplier active)
# ---------------------------------------------------------------------------

class TestLocateDataSetANegX5:
    """Tests for LocateDataSetA with neg_x_multiplier set (config 003).

    The multiplier scales negative-row counts but NOT positive-row counts,
    so test_positive_rows asserts the same values as the default class.
    """

    def test_positive_rows(
        self, dataset_config_003, dataset_input_003, dataset_select_003
    ):
        """Positive rows are unaffected by neg_x_multiplier."""
        ds = LocateDataSetA(
            dataset_config_003,
            input_data=dataset_input_003.input_data,
            selected_profiles=dataset_select_003.selected_profiles,
        )
        for tgt in TARGETS:
            ds.select_positive_rows(tgt, TARGET_VALUES[tgt])

        expected_pos = {"temp": 12, "psal": 18, "pres": 9}
        for tgt in TARGETS:
            assert isinstance(ds.positive_rows[tgt], pl.DataFrame)
            assert ds.positive_rows[tgt].shape[0] == expected_pos[tgt]
            assert ds.positive_rows[tgt].shape[1] == 9

    def test_negative_rows(
        self, dataset_config_003, dataset_input_003, dataset_select_003
    ):
        """Negative rows are scaled up by the neg_x_multiplier."""
        ds = LocateDataSetA(
            dataset_config_003,
            input_data=dataset_input_003.input_data,
            selected_profiles=dataset_select_003.selected_profiles,
        )
        for tgt in TARGETS:
            ds.select_positive_rows(tgt, TARGET_VALUES[tgt])
            ds.select_negative_rows(tgt, TARGET_VALUES[tgt])

        expected_neg = {"temp": 165, "psal": 231, "pres": 120}
        for tgt in TARGETS:
            assert isinstance(ds.negative_rows[tgt], pl.DataFrame)
            assert ds.negative_rows[tgt].shape[0] == expected_neg[tgt]
            assert ds.negative_rows[tgt].shape[1] == 8

    def test_selected_rows(
        self, dataset_config_003, dataset_input_003, dataset_select_003
    ):
        """process_targets aggregates the multiplied negative rows."""
        ds = LocateDataSetA(
            dataset_config_003,
            input_data=dataset_input_003.input_data,
            selected_profiles=dataset_select_003.selected_profiles,
        )
        ds.process_targets()

        expected = {"temp": 177, "psal": 249, "pres": 129}
        for tgt in TARGETS:
            assert isinstance(ds.selected_rows[tgt], pl.DataFrame)
            assert ds.selected_rows[tgt].shape[0] == expected[tgt]
            assert ds.selected_rows[tgt].shape[1] == 9

    def test_write_selected_rows(
        self, dataset_config_003, dataset_input_003, dataset_select_003, test_output_dir
    ):
        """write_selected_rows (NegX5) produces a parquet per target."""
        ds = LocateDataSetA(
            dataset_config_003,
            input_data=dataset_input_003.input_data,
            selected_profiles=dataset_select_003.selected_profiles,
        )
        output_paths = {
            tgt: str(test_output_dir / f"test_selected_rows_negx5_{tgt}.parquet")
            for tgt in TARGETS
        }
        for tgt in TARGETS:
            ds.output_file_names[tgt] = output_paths[tgt]

        ds.process_targets()
        ds.write_selected_rows()

        for tgt in TARGETS:
            assert os.path.exists(output_paths[tgt])
            os.remove(output_paths[tgt])  # comment out to debug