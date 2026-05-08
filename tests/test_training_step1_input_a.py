"""
This module contains unit tests for the `InputTrainingSetA` class, focusing on its
ability to correctly initialize from a `TrainingConfig` and load training and test
datasets using Polars.
"""

import unittest
from pathlib import Path

import polars as pl

from aiqclib.common.config.training_config import TrainingConfig
from aiqclib.train.step1_read_input.dataset_a import InputTrainingSetA


class TestInputTrainingSetA(unittest.TestCase):
    """
    A suite of tests ensuring that the InputTrainingSetA class
    correctly loads and processes training data sets.
    """

    def setUp(self):
        """
        Prepare a TrainingConfig instance pointing to a test YAML file with
        specified training sets. Also defines file paths where training
        and test data can be loaded from.
        """
        self.config_file_path = (
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_training_001.yaml"
        )
        self.config = TrainingConfig(str(self.config_file_path))
        self.config.select("NRT_BO_001")
        data_path = Path(__file__).resolve().parent / "data" / "training"
        self.input_file_names = {
            "train": {
                "temp": str(data_path / "train_set_temp.parquet"),
                "psal": str(data_path / "train_set_psal.parquet"),
                "pres": str(data_path / "train_set_pres.parquet"),
            },
            "test": {
                "temp": str(data_path / "test_set_temp.parquet"),
                "psal": str(data_path / "test_set_psal.parquet"),
                "pres": str(data_path / "test_set_pres.parquet"),
            },
        }

    def test_step_name(self):
        """Check that the step name in the InputTrainingSetA instance is 'input'."""
        ds = InputTrainingSetA(self.config)
        self.assertEqual(ds.step_name, "input")

    def test_input_file_names(self):
        """
        Verify that file names for training and test sets are correctly resolved
        from the training configuration.
        """
        ds = InputTrainingSetA(self.config)
        self.assertEqual(
            "/path/to/input_1/nrt_bo_001/input_folder_1/train_set_temp.parquet",
            str(ds.input_file_names["train"]["temp"]),
        )
        self.assertEqual(
            "/path/to/input_1/nrt_bo_001/input_folder_1/train_set_psal.parquet",
            str(ds.input_file_names["train"]["psal"]),
        )
        self.assertEqual(
            "/path/to/input_1/nrt_bo_001/input_folder_1/train_set_pres.parquet",
            str(ds.input_file_names["train"]["pres"]),
        )
        self.assertEqual(
            "/path/to/input_1/nrt_bo_001/input_folder_1/test_set_temp.parquet",
            str(ds.input_file_names["test"]["temp"]),
        )
        self.assertEqual(
            "/path/to/input_1/nrt_bo_001/input_folder_1/test_set_psal.parquet",
            str(ds.input_file_names["test"]["psal"]),
        )

        self.assertEqual(
            "/path/to/input_1/nrt_bo_001/input_folder_1/test_set_pres.parquet",
            str(ds.input_file_names["test"]["pres"]),
        )

    def test_read_files(self):
        """
        Confirm that training and test data sets are loaded into
        Polars DataFrame objects, with expected shapes for each variable.
        """
        ds = InputTrainingSetA(self.config)
        # Override input_file_names to point to actual local test data files
        # for reading, as the config paths might be generic/placeholder.
        ds.input_file_names = self.input_file_names

        ds.process_targets()

        self.assertIsInstance(ds.training_sets["temp"], pl.DataFrame)
        self.assertEqual(ds.training_sets["temp"].shape[0], 116)
        self.assertEqual(ds.training_sets["temp"].shape[1], 57)

        self.assertIsInstance(ds.test_sets["temp"], pl.DataFrame)
        self.assertEqual(ds.test_sets["temp"].shape[0], 12)
        self.assertEqual(ds.test_sets["temp"].shape[1], 56)

        self.assertIsInstance(ds.training_sets["psal"], pl.DataFrame)
        self.assertEqual(ds.training_sets["psal"].shape[0], 126)
        self.assertEqual(ds.training_sets["psal"].shape[1], 57)

        self.assertIsInstance(ds.test_sets["psal"], pl.DataFrame)
        self.assertEqual(ds.test_sets["psal"].shape[0], 14)
        self.assertEqual(ds.test_sets["psal"].shape[1], 56)

        self.assertIsInstance(ds.training_sets["pres"], pl.DataFrame)
        self.assertEqual(ds.training_sets["pres"].shape[0], 110)
        self.assertEqual(ds.training_sets["pres"].shape[1], 57)

        self.assertIsInstance(ds.test_sets["pres"], pl.DataFrame)
        self.assertEqual(ds.test_sets["pres"].shape[0], 12)
        self.assertEqual(ds.test_sets["pres"].shape[1], 56)

    def test_read_training_set_incorrect_file_names(self):
        """
        Verify that `process_targets` raises a `FileNotFoundError` if *any* of the
        configured training or test file paths do not exist when initialized from config.
        """
        ds = InputTrainingSetA(self.config)
        with self.assertRaises(FileNotFoundError):
            ds.process_targets()

    def test_read_test_set_incorrect_file_names(self):
        """
        Verify that `process_targets` raises a `FileNotFoundError` if the *test*
        set file paths do not exist, even if training set paths are valid.
        """
        ds = InputTrainingSetA(self.config)
        ds.input_file_names["train"] = self.input_file_names["train"]
        with self.assertRaises(FileNotFoundError):
            ds.process_targets()
