"""Unit tests for the ``InputTrainingSetA`` class.

InputTrainingSetA loads pre-split training and test parquets (produced by
``SplitDataSetA``) and exposes them as ``training_sets`` / ``test_sets``
keyed by target name.

Refactored from a single ``unittest.TestCase`` class. Bridges from the
prepare stage (test_prepare_*) to the training stage (test_training_*);
uses ``training_config_001`` from conftest plus a small inline
``input_file_names`` dict.
"""

import polars as pl
import pytest

from aiqclib.train.step1_read_input.dataset_a import InputTrainingSetA

from tests.conftest import TARGETS


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _build_input_file_names(training_dir):
    """Build the ``input_file_names`` dict pointing at tests/data/training/.

    Same shape as the legacy setUp's self.input_file_names — keyed by
    ``"train"``/``"test"`` then by target name.
    """
    return {
        kind: {tgt: str(training_dir / f"{kind}_set_{tgt}.parquet") for tgt in TARGETS}
        for kind in ("train", "test")
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestInputTrainingSetA:
    """Tests for InputTrainingSetA: identity, file-name resolution, and read."""

    def test_step_name(self, training_config_001):
        """step_name == 'input'."""
        ds = InputTrainingSetA(training_config_001)
        assert ds.step_name == "input"

    def test_input_file_names(self, training_config_001):
        """Default per-(kind, target) input paths come from config.path_info."""
        ds = InputTrainingSetA(training_config_001)
        base = "/path/to/input_1/nrt_bo_001/input_folder_1"
        for kind in ("train", "test"):
            for tgt in TARGETS:
                assert (
                    str(ds.input_file_names[kind][tgt])
                    == f"{base}/{kind}_set_{tgt}.parquet"
                )

    def test_read_files(self, training_config_001, training_dir):
        """process_targets loads each (kind, target) parquet with expected shape."""
        ds = InputTrainingSetA(training_config_001)
        # Override the config-derived paths with paths to actual fixture files.
        ds.input_file_names = _build_input_file_names(training_dir)
        ds.process_targets()

        # TODO: update per-target rows after data reduction.
        # Was train=(116, 126, 110) with 57 cols, test=(12, 14, 12) with 56 cols.
        expected_train_rows = {"temp": 22, "psal": 34, "pres": 18}
        expected_test_rows = {"temp": 2, "psal": 2, "pres": 0}
        for tgt in TARGETS:
            assert isinstance(ds.training_sets[tgt], pl.DataFrame)
            assert ds.training_sets[tgt].shape[0] == expected_train_rows[tgt]
            assert ds.training_sets[tgt].shape[1] == 57

            assert isinstance(ds.test_sets[tgt], pl.DataFrame)
            assert ds.test_sets[tgt].shape[0] == expected_test_rows[tgt]
            assert ds.test_sets[tgt].shape[1] == 56

    def test_read_training_set_incorrect_file_names(self, training_config_001):
        """process_targets with bogus (config-derived) paths raises FileNotFoundError.

        The config's path_info points at /path/to/... which doesn't exist on
        disk; without an override, process_targets fails on the first read.
        """
        ds = InputTrainingSetA(training_config_001)
        with pytest.raises(FileNotFoundError):
            ds.process_targets()

    def test_read_test_set_incorrect_file_names(self, training_config_001, training_dir):
        """When train paths are valid but test paths aren't, process_targets fails.

        Sets train paths only; leaves test paths at their config-derived
        /path/to/... default. process_targets reads train first (succeeds),
        then test (fails with FileNotFoundError).
        """
        ds = InputTrainingSetA(training_config_001)
        ds.input_file_names["train"] = _build_input_file_names(training_dir)["train"]
        with pytest.raises(FileNotFoundError):
            ds.process_targets()