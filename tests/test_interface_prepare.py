"""Unit tests for the ``create_training_dataset`` interface function.

Runs the full prepare pipeline (summary → select → locate → extract →
split) against test configs and verifies that the expected output folder
structure and per-target files are produced.

Refactored from two classes (``TestCreateTrainingDataSet`` parametrized over
3 configs, plus ``TestCreateTrainingDataSetNegX5`` for config 003) into
the same shape but with conftest fixtures, per-target file-existence loops,
and the path_info-override pattern factored into a small helper.
"""

import shutil

import pytest

from aiqclib.interface.prepare import create_training_dataset

from tests.conftest import TARGETS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wire_path_info(config, test_output_dir, input_dir, extra=None):
    """Override the config's path_info to redirect output to test_output_dir.

    The default ``input.step_folder_name=""`` means the input file is found
    directly under ``input_dir`` (i.e. ``tests/data/input/``). Pass ``extra``
    as a dict to merge additional path_info entries (e.g. the NegX5 variant
    sets a custom ``split.step_folder_name="training"``).
    """
    config.data["input_file_name"] = "nrt_cora_bo_test.parquet"
    path_info = {
        "name": "data_set_1",
        "common": {"base_path": str(test_output_dir)},
        "input": {"base_path": str(input_dir), "step_folder_name": ""},
    }
    if extra:
        path_info.update(extra)
    config.data["path_info"] = path_info


def _assert_prepare_outputs(output_folder, *, split_dir_name="split"):
    """Assert every expected output file from create_training_dataset exists.

    ``split_dir_name`` defaults to "split" but the NegX5 variant uses
    "training" via a custom ``split.step_folder_name`` in path_info.
    """
    dir_summary = output_folder / "summary"
    dir_select = output_folder / "select"
    dir_locate = output_folder / "locate"
    dir_extract = output_folder / "extract"
    dir_split = output_folder / split_dir_name

    assert (dir_summary / "summary_stats.tsv").exists()
    assert (dir_select / "selected_profiles.parquet").exists()
    for tgt in TARGETS:
        assert (dir_locate / f"selected_rows_{tgt}.parquet").exists()
        assert (dir_extract / f"extracted_features_{tgt}.parquet").exists()
        assert (dir_split / f"train_set_{tgt}.parquet").exists()
        assert (dir_split / f"test_set_{tgt}.parquet").exists()


def _cleanup_output_folder(config, test_output_dir):
    """Remove the config's ``dataset_folder_name`` directory if present."""
    output_folder = test_output_dir / config.data["dataset_folder_name"]
    if output_folder.exists() and output_folder.is_dir():
        shutil.rmtree(output_folder)


# ---------------------------------------------------------------------------
# Three-config tests (default behaviour, no custom step folder names)
# ---------------------------------------------------------------------------


class TestCreateTrainingDataSet:
    """create_training_dataset against configs 001, 004, 005.

    Each config produces a fully-populated output folder tree. Tests
    parametrize over ``idx ∈ {0, 1, 2}`` to run each in isolation, with
    cleanup happening per-test via the autouse fixture.
    """

    @pytest.fixture(autouse=True)
    def setup_and_cleanup(
        self,
        dataset_config_001,
        dataset_config_004,
        dataset_config_005,
        test_output_dir,
        input_dir,
    ):
        """Wire the three configs and clean up generated folders afterwards."""
        self.configs = [dataset_config_001, dataset_config_004, dataset_config_005]
        self.test_output_dir = test_output_dir
        for c in self.configs:
            _wire_path_info(c, test_output_dir, input_dir)

        yield

        for c in self.configs:
            _cleanup_output_folder(c, test_output_dir)

    @pytest.mark.parametrize("idx", range(3))
    def test_create_training_data_set(self, idx):
        """End-to-end prepare pipeline produces all expected outputs."""
        create_training_dataset(self.configs[idx])

        output_folder = (
            self.test_output_dir / self.configs[idx].data["dataset_folder_name"]
        )
        _assert_prepare_outputs(output_folder)


# ---------------------------------------------------------------------------
# NegX5 variant (config 003 with custom split folder name)
# ---------------------------------------------------------------------------


class TestCreateTrainingDataSetNegX5:
    """create_training_dataset against config 003 with a custom split folder.

    The NegX5 variant sets ``split.step_folder_name="training"`` to verify
    that the configured folder override flows through correctly. Train/test
    parquets land in ``training/`` rather than the default ``split/``.
    """

    @pytest.fixture(autouse=True)
    def setup_and_cleanup(self, dataset_config_003, test_output_dir, input_dir):
        """Wire config 003 with the custom split-folder override; clean up afterwards."""
        self.config = dataset_config_003
        self.test_output_dir = test_output_dir
        _wire_path_info(
            self.config,
            test_output_dir,
            input_dir,
            extra={"split": {"step_folder_name": "training"}},
        )

        yield

        _cleanup_output_folder(self.config, test_output_dir)

    def test_create_training_data_set(self):
        """End-to-end with custom split folder produces train/test in training/."""
        create_training_dataset(self.config)

        output_folder = self.test_output_dir / self.config.data["dataset_folder_name"]
        _assert_prepare_outputs(output_folder, split_dir_name="training")
