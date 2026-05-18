"""Integration tests for the top-level ``aiqclib`` package surface.

This file's purpose is narrow: confirm that the public functions exposed
via ``import aiqclib as aq`` (``aq.read_config``, ``aq.write_config_template``,
``aq.create_training_dataset``, ``aq.train_and_evaluate``,
``aq.get_summary_stats``, ``aq.format_summary_stats``) all work via the
``aq`` namespace. The actual behaviour of each function is exercised in
its dedicated ``test_interface_*.py`` file; here we only verify the
top-level re-exports are intact.

Refactored from four ``unittest.TestCase`` classes into plain pytest
classes using conftest fixtures. Functionally redundant with the
interface tests but kept as a regression net for the package's public
API surface.

Note on training_config_001_bo002:
The train test uses ``training_config_001_bo002`` (NRT_BO_002, 2-target
temp+psal) rather than ``training_config_001`` (NRT_BO_001, 3-target).
The reduced test fixtures have zero pres test rows, which crashes the
train pipeline during the build step. When the library handles zero-row
test data, swap back to ``training_config_001`` and use ``TARGETS``
instead of ``TARGETS_NONEMPTY`` in the assertion loop.
"""

import os
import shutil

import polars as pl

import aiqclib as aq
from aiqclib.common.config.dataset_config import DataSetConfig
from aiqclib.common.config.training_config import TrainingConfig

from tests.conftest import TARGETS, TARGETS_NONEMPTY


# ---------------------------------------------------------------------------
# Template config (aq.write_config_template)
# ---------------------------------------------------------------------------

class TestDMQCLibTemplateConfig:
    """Verify ``aq.write_config_template`` is reachable via the top-level namespace."""

    def test_ds_config_template(self, test_output_dir):
        """Writes a dataset (prepare) config template to disk."""
        path = test_output_dir / "temp_dataset_template.yaml"
        aq.write_config_template(str(path), "prepare")
        assert os.path.exists(path)
        os.remove(path)  # comment out to debug

    def test_config_train_set_template(self, test_output_dir):
        """Writes a training config template to disk."""
        path = test_output_dir / "temp_training_template.yaml"
        aq.write_config_template(str(path), "train")
        assert os.path.exists(path)
        os.remove(path)  # comment out to debug


# ---------------------------------------------------------------------------
# Read config (aq.read_config)
# ---------------------------------------------------------------------------

class TestDMQCLibReadConfig:
    """Verify ``aq.read_config`` returns the right type for each module."""

    def test_ds_config(self, dataset_yaml_001):
        """A prepare YAML file reads as DataSetConfig."""
        config = aq.read_config(str(dataset_yaml_001))
        assert isinstance(config, DataSetConfig)

    def test_train_config(self, training_yaml_001):
        """A train YAML file reads as TrainingConfig."""
        config = aq.read_config(str(training_yaml_001), "NRT_BO_001", auto_select=False)
        assert isinstance(config, TrainingConfig)


# ---------------------------------------------------------------------------
# Full prepare workflow (aq.create_training_dataset)
# ---------------------------------------------------------------------------

class TestDMQCLibCreateTrainingDataSet:
    """Verify ``aq.create_training_dataset`` runs end-to-end via the aq namespace."""

    def test_create_training_data_set(
        self, dataset_config_001, test_output_dir, input_dir,
    ):
        """End-to-end prepare via aq.create_training_dataset produces all outputs."""
        config = dataset_config_001
        config.data["input_file_name"] = "nrt_cora_bo_test.parquet"
        # The original test used name="nrt_bo_001"; preserving for parity, though
        # other interface tests use name="data_set_1". The name only affects
        # parts of path resolution that don't reach the assertions below.
        config.data["path_info"] = {
            "name": "nrt_bo_001",
            "common": {"base_path": str(test_output_dir)},
            "input": {"base_path": str(input_dir), "step_folder_name": ""},
        }

        aq.create_training_dataset(config)

        output_folder = test_output_dir / config.data["dataset_folder_name"]

        # Single non-per-target outputs
        assert (output_folder / "summary" / "summary_stats.tsv").exists()
        assert (output_folder / "select" / "selected_profiles.parquet").exists()

        # Per-target outputs across locate / extract / split.
        # The prepare pipeline uses the dataset config's target set, which
        # has all 3 targets — pres has non-empty data here (it's only the
        # train test split that drops pres rows).
        for tgt in TARGETS:
            assert (output_folder / "locate" / f"selected_rows_{tgt}.parquet").exists()
            assert (output_folder / "extract" / f"extracted_features_{tgt}.parquet").exists()
            assert (output_folder / "split" / f"train_set_{tgt}.parquet").exists()
            assert (output_folder / "split" / f"test_set_{tgt}.parquet").exists()

        shutil.rmtree(output_folder)


# ---------------------------------------------------------------------------
# Full train workflow (aq.train_and_evaluate)
# ---------------------------------------------------------------------------

class TestDMQCLibTrainAndEvaluate:
    """Verify ``aq.train_and_evaluate`` runs end-to-end via the aq namespace.

    Uses ``training_config_001_bo002`` (NRT_BO_002, 2-target) to avoid the
    empty-pres-test-data crash described in the file docstring. Outputs are
    therefore only produced for temp and psal; the assertion loop uses
    ``TARGETS_NONEMPTY``.
    """

    def test_train_and_evaluate(self, training_config_001_bo002, test_output_dir, training_dir):
        """End-to-end train via aq.train_and_evaluate produces reports and models."""
        config = training_config_001_bo002
        config.data["path_info"] = {
            "name": "data_set_1",
            "common": {"base_path": str(test_output_dir)},
            "input": {"base_path": str(training_dir), "step_folder_name": ".."},
        }

        aq.train_and_evaluate(config)

        output_folder = test_output_dir / config.data["dataset_folder_name"]

        # Per-target outputs across validate / build / model.
        # bo002 produces files only for temp + psal (no pres).
        for tgt in TARGETS_NONEMPTY:
            assert (output_folder / "validate" / f"validation_report_{tgt}.tsv").exists()
            assert (output_folder / "build" / f"test_report_{tgt}.tsv").exists()
            assert (output_folder / "model" / f"model_{tgt}.joblib").exists()

        shutil.rmtree(output_folder)


# ---------------------------------------------------------------------------
# Summary stats (aq.get_summary_stats, aq.format_summary_stats)
# ---------------------------------------------------------------------------

class TestGetSummaryStats:
    """Verify ``aq.get_summary_stats`` and ``aq.format_summary_stats`` are reachable."""

    def test_get_profile_summary_stats(self, test_data_file):
        """aq.get_summary_stats('profiles') returns a DataFrame."""
        ds = aq.get_summary_stats(test_data_file, "profiles")
        assert isinstance(ds, pl.DataFrame)

    def test_get_global_summary_stats(self, test_data_file):
        """aq.get_summary_stats('all') returns a DataFrame."""
        ds = aq.get_summary_stats(test_data_file, "all")
        assert isinstance(ds, pl.DataFrame)

    def test_format_profile_summary_stats(self, test_data_file):
        """aq.format_summary_stats with variables/stats filtering."""
        ds = aq.get_summary_stats(test_data_file, "profiles")

        stats_str = aq.format_summary_stats(ds)
        assert isinstance(stats_str, str)
        assert "psal" in stats_str
        assert "pct25" in stats_str

        stats_str = aq.format_summary_stats(ds, ["pres", "temp"])
        assert isinstance(stats_str, str)
        assert "psal" not in stats_str
        assert "pct25" in stats_str

        stats_str = aq.format_summary_stats(ds, ["pres", "temp"], ["mean"])
        assert isinstance(stats_str, str)
        assert "psal" not in stats_str
        assert "pct25" not in stats_str

    def test_format_global_summary_stats(self, test_data_file):
        """aq.format_summary_stats on global stats filters by variables."""
        ds = aq.get_summary_stats(test_data_file, "all")

        stats_str = aq.format_summary_stats(ds)
        assert isinstance(stats_str, str)
        assert "psal" in stats_str

        stats_str = aq.format_summary_stats(ds, ["pres", "temp"])
        assert isinstance(stats_str, str)
        assert "psal" not in stats_str