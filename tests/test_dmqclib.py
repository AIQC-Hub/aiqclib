"""Integration tests for aiqclib.

This module contains integration tests that exercise various aiqclib functionality,
including:

1. Creating dataset and training configuration templates.
2. Reading existing configuration files.
3. Creating a training dataset for Argo-based data (prepare workflow).
4. Training and evaluating models (train workflow).
5. Testing summary statistics utilities.
"""

import os
import shutil
import unittest
from pathlib import Path

import polars as pl

import aiqclib as aq
from aiqclib.common.config.dataset_config import DataSetConfig
from aiqclib.common.config.training_config import TrainingConfig


class TestDMQCLibTemplateConfig(unittest.TestCase):
    """Tests for creating dataset and training configuration templates
    using the aiqclib library.
    """

    def setUp(self):
        """Prepare file paths for dataset and training configuration templates.

        These temporary files will be written and removed during testing.
        """
        self.ds_config_template_file = str(
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "temp_dataset_template.yaml"
        )
        self.config_train_set_template_file = str(
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "temp_training_template.yaml"
        )

    def test_ds_config_template(self):
        """Verify that a dataset (prepare) configuration template can be written and removed."""
        aq.write_config_template(self.ds_config_template_file, "prepare")
        self.assertTrue(os.path.exists(self.ds_config_template_file))
        os.remove(self.ds_config_template_file)

    def test_config_train_set_template(self):
        """Verify that a training configuration template can be written and removed."""
        aq.write_config_template(self.config_train_set_template_file, "train")
        self.assertTrue(os.path.exists(self.config_train_set_template_file))
        os.remove(self.config_train_set_template_file)


class TestDMQCLibReadConfig(unittest.TestCase):
    """Tests for reading dataset (prepare) and training (train) configuration files
    using the aiqclib library.
    """

    def setUp(self):
        """Define paths to existing dataset and training configuration files.

        These files are used in subsequent read tests.
        """
        self.ds_config_file = str(
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_dataset_001.yaml"
        )
        self.train_config_file = str(
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_training_001.yaml"
        )

    def test_ds_config(self):
        """Verify that reading a dataset configuration returns a DataSetConfig instance."""
        config = aq.read_config(self.ds_config_file)
        self.assertIsInstance(config, DataSetConfig)

    def test_train_config(self):
        """Verify that reading a training configuration returns a TrainingConfig instance."""
        config = aq.read_config(self.train_config_file)
        self.assertIsInstance(config, TrainingConfig)


class TestDMQCLibCreateTrainingDataSet(unittest.TestCase):
    """Tests for creating a training dataset (prepare workflow).

    Ensures that all expected intermediate and final outputs are generated.
    """

    def setUp(self):
        """Load dataset configuration, specify file names and paths,
        and prepare test output directories.
        """
        self.config_file_path = str(
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_dataset_001.yaml"
        )
        self.config = DataSetConfig(str(self.config_file_path))
        self.config.select("NRT_BO_001")
        self.config.data["input_file_name"] = "nrt_cora_bo_test.parquet"
        self.test_data_location = Path(__file__).resolve().parent / "data" / "test"
        self.input_data_path = Path(__file__).resolve().parent / "data" / "input"
        self.config.data["path_info"] = {
            "name": "nrt_bo_001",
            "common": {"base_path": str(self.test_data_location)},
            "input": {"base_path": str(self.input_data_path), "step_folder_name": ""},
        }

    def test_create_training_data_set(self):
        """Run the full 'prepare' workflow via `aq.create_training_dataset`.

        Then, confirm that all expected output files and folders are created.
        """
        aq.create_training_dataset(self.config)

        output_folder = (
            self.test_data_location / self.config.data["dataset_folder_name"]
        )

        self.assertTrue(
            os.path.exists(str(output_folder / "summary" / "summary_stats.tsv"))
        )
        self.assertTrue(
            os.path.exists(str(output_folder / "select" / "selected_profiles.parquet"))
        )
        self.assertTrue(
            os.path.exists(str(output_folder / "locate" / "selected_rows_temp.parquet"))
        )
        self.assertTrue(
            os.path.exists(str(output_folder / "locate" / "selected_rows_psal.parquet"))
        )
        self.assertTrue(
            os.path.exists(str(output_folder / "locate" / "selected_rows_pres.parquet"))
        )
        self.assertTrue(
            os.path.exists(
                str(output_folder / "extract" / "extracted_features_temp.parquet")
            )
        )
        self.assertTrue(
            os.path.exists(
                str(output_folder / "extract" / "extracted_features_psal.parquet")
            )
        )
        self.assertTrue(
            os.path.exists(
                str(output_folder / "extract" / "extracted_features_pres.parquet")
            )
        )
        self.assertTrue(
            os.path.exists(str(output_folder / "split" / "train_set_temp.parquet"))
        )
        self.assertTrue(
            os.path.exists(str(output_folder / "split" / "train_set_psal.parquet"))
        )
        self.assertTrue(
            os.path.exists(str(output_folder / "split" / "train_set_pres.parquet"))
        )
        self.assertTrue(
            os.path.exists(str(output_folder / "split" / "test_set_temp.parquet"))
        )
        self.assertTrue(
            os.path.exists(str(output_folder / "split" / "test_set_psal.parquet"))
        )
        self.assertTrue(
            os.path.exists(str(output_folder / "split" / "test_set_pres.parquet"))
        )

        shutil.rmtree(output_folder)


class TestDMQCLibTrainAndEvaluate(unittest.TestCase):
    """Tests for the training workflow.

    This suite ensures training and evaluation produce expected validation/test
    results and model files.
    """

    def setUp(self):
        """Load a TrainingConfig, define input and output paths,
        and prepare directories for the train-and-evaluate steps.
        """
        self.config_file_path = (
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_training_001.yaml"
        )
        self.config = TrainingConfig(str(self.config_file_path))
        self.config.select("NRT_BO_001")
        self.test_data_location = Path(__file__).resolve().parent / "data" / "test"
        self.input_data_path = Path(__file__).resolve().parent / "data" / "training"
        self.config.data["path_info"] = {
            "name": "data_set_1",
            "common": {"base_path": str(self.test_data_location)},
            "input": {
                "base_path": str(self.input_data_path),
                "step_folder_name": "..",
            },
        }

    def test_train_and_evaluate(self):
        """Run the `aq.train_and_evaluate` function with the loaded configuration.

        Then, verify that validation reports, test reports, and trained model
        files are successfully generated.
        """
        aq.train_and_evaluate(self.config)

        output_folder = (
            self.test_data_location / self.config.data["dataset_folder_name"]
        )

        self.assertTrue(
            os.path.exists(
                str(output_folder / "validate" / "validation_report_temp.tsv")
            )
        )
        self.assertTrue(
            os.path.exists(
                str(output_folder / "validate" / "validation_report_psal.tsv")
            )
        )
        self.assertTrue(
            os.path.exists(
                str(output_folder / "validate" / "validation_report_pres.tsv")
            )
        )
        self.assertTrue(
            os.path.exists(str(output_folder / "build" / "test_report_temp.tsv"))
        )
        self.assertTrue(
            os.path.exists(str(output_folder / "build" / "test_report_psal.tsv"))
        )
        self.assertTrue(
            os.path.exists(str(output_folder / "build" / "test_report_pres.tsv"))
        )
        self.assertTrue(
            os.path.exists(str(output_folder / "model" / "model_temp.joblib"))
        )
        self.assertTrue(
            os.path.exists(str(output_folder / "model" / "model_psal.joblib"))
        )
        self.assertTrue(
            os.path.exists(str(output_folder / "model" / "model_pres.joblib"))
        )

        shutil.rmtree(output_folder)


class TestGetSummaryStats(unittest.TestCase):
    """Tests for the summary statistics utility functions in aiqclib.

    This suite verifies the correct calculation and formatting of summary
    statistics from a dataset.
    """

    def setUp(self):
        """Set up the test environment by defining the sample input file path."""
        self.test_data_file = (
            Path(__file__).resolve().parent
            / "data"
            / "input"
            / "nrt_cora_bo_test.parquet"
        )

    def test_get_profile_summary_stats(self):
        """Verify that `aq.get_summary_stats` returns a DataFrame with profile-level summaries."""
        ds = aq.get_summary_stats(self.test_data_file, "profiles")
        self.assertIsInstance(ds, pl.DataFrame)

    def test_get_global_summary_stats(self):
        """Verify that `aq.get_summary_stats` returns a DataFrame with global summaries."""
        ds = aq.get_summary_stats(self.test_data_file, "all")
        self.assertIsInstance(ds, pl.DataFrame)

    def test_format_profile_summary_stats(self):
        """Verify that `aq.format_summary_stats` correctly formats profile-level summaries.

        Checks for presence/absence of variables and statistics based on filters.
        """
        ds = aq.get_summary_stats(self.test_data_file, "profiles")

        stats_dict = aq.format_summary_stats(ds)
        self.assertIsInstance(stats_dict, str)
        self.assertIn("psal", stats_dict)
        self.assertIn("pct25", stats_dict)

        stats_dict = aq.format_summary_stats(ds, ["pres", "temp"])
        self.assertIsInstance(stats_dict, str)
        self.assertNotIn("psal", stats_dict)
        self.assertIn("pct25", stats_dict)

        stats_dict = aq.format_summary_stats(ds, ["pres", "temp"], ["mean"])
        self.assertIsInstance(stats_dict, str)
        self.assertNotIn("psal", stats_dict)
        self.assertNotIn("pct25", stats_dict)

    def test_format_global_summary_stats(self):
        """Verify that `aq.format_summary_stats` correctly formats global summaries.

        Checks for presence/absence of variables based on filters.
        """
        ds = aq.get_summary_stats(self.test_data_file, "all")

        stats_dict = aq.format_summary_stats(ds)
        self.assertIsInstance(stats_dict, str)
        self.assertIn("psal", stats_dict)

        stats_dict = aq.format_summary_stats(ds, ["pres", "temp"])
        self.assertIsInstance(stats_dict, str)
        self.assertNotIn("psal", stats_dict)
