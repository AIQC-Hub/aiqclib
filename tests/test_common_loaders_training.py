"""
This module contains unit tests for verifying the correct loading of various
training-related classes via the aiqclib.common.loader.training_loader functions.
It ensures that the appropriate input, model validation, and model build classes
are instantiated and configured based on the provided training configuration.
"""

import unittest
from pathlib import Path

import polars as pl

from aiqclib.common.config.training_config import TrainingConfig
from aiqclib.common.loader.training_loader import (
    load_step1_input_training_set,
    load_step2_model_validation_class,
    load_step4_build_model_class,
)
from aiqclib.train.step1_read_input.dataset_a import InputTrainingSetA
from aiqclib.train.step2_validate_model.kfold_validation import KFoldValidation
from aiqclib.train.step2_validate_model.kfold_validation_suite import (
    KFoldValidationSuite,
)
from aiqclib.train.step4_build_model.build_model import BuildModel
from aiqclib.train.step4_build_model.build_model_suite import BuildModelSuite


class TestTrainingInputClassLoader(unittest.TestCase):
    """
    Tests for verifying that the correct input training class
    (InputTrainingSetA) is loaded from the config.
    """

    def setUp(self):
        """
        Initialize a training configuration object and select a dataset
        for input loading tests.
        """
        self.config_file_path = (
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_training_001.yaml"
        )
        self.config = TrainingConfig(str(self.config_file_path))
        self.config.select("NRT_BO_001")

    def test_load_dataset_valid_config(self):
        """
        Check that load_step1_input_training_set returns an InputTrainingSetA
        instance with the expected step name.
        """
        ds = load_step1_input_training_set(self.config)
        self.assertIsInstance(ds, InputTrainingSetA)
        self.assertEqual(ds.step_name, "input")

    def test_load_input_class_with_invalid_config(self):
        """
        Ensure that invalid input class name raises a ValueError.
        """
        self.config.data["step_class_set"]["steps"]["input"] = "InvalidClass"
        with self.assertRaises(ValueError):
            _ = load_step1_input_training_set(self.config)


class TestModelValidationClassLoader(unittest.TestCase):
    """
    Tests confirming that the correct model validation class
    (KFoldValidation) is loaded and configured.
    """

    def setUp(self):
        """
        Initialize a training configuration and load input data for testing
        the validation class loader.
        """
        self.config_file_path = str(
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

        self.ds_input = load_step1_input_training_set(self.config)
        self.ds_input.input_file_names = self.input_file_names
        self.ds_input.process_targets()

    def test_load_dataset_valid_config(self):
        """
        Check that load_step2_model_validation_class returns a KFoldValidation
        instance with the expected step name.
        """
        ds = load_step2_model_validation_class(self.config)
        self.assertIsInstance(ds, KFoldValidation)
        self.assertEqual(ds.step_name, "validate")

    def test_load_validation_suite_dataset(self):
        """
        Check that load_step2_model_validation_class returns a KFoldValidationSuite
        instance with the expected step name.
        """
        self.config.data["step_class_set"]["steps"]["model"] = "ModelSuite"
        self.config.data["step_class_set"]["steps"]["validate"] = "KFoldValidationSuite"
        ds = load_step2_model_validation_class(self.config)
        self.assertIsInstance(ds, KFoldValidationSuite)
        self.assertEqual(ds.step_name, "validate")

    def test_training_set_data(self):
        """
        Ensure that when training_sets are provided, KFoldValidation
        is instantiated with the correct Polars DataFrame sizes.
        """
        ds = load_step2_model_validation_class(self.config, self.ds_input.training_sets)

        self.assertIsInstance(ds.training_sets["temp"], pl.DataFrame)
        self.assertEqual(ds.training_sets["temp"].shape[0], 116)
        self.assertEqual(ds.training_sets["temp"].shape[1], 57)

        self.assertIsInstance(ds.training_sets["psal"], pl.DataFrame)
        self.assertEqual(ds.training_sets["psal"].shape[0], 126)
        self.assertEqual(ds.training_sets["psal"].shape[1], 57)

        self.assertIsInstance(ds.training_sets["pres"], pl.DataFrame)
        self.assertEqual(ds.training_sets["pres"].shape[0], 110)
        self.assertEqual(ds.training_sets["pres"].shape[1], 57)

    def test_suit_training_set_data(self):
        """
        Ensure that when training_sets are provided, KFoldValidationSuite
        is instantiated with the correct Polars DataFrame sizes.
        """
        self.config.data["step_class_set"]["steps"]["model"] = "ModelSuite"
        self.config.data["step_class_set"]["steps"]["validate"] = "KFoldValidationSuite"
        ds = load_step2_model_validation_class(self.config, self.ds_input.training_sets)

        self.assertIsInstance(ds.training_sets["temp"], pl.DataFrame)
        self.assertEqual(ds.training_sets["temp"].shape[0], 116)
        self.assertEqual(ds.training_sets["temp"].shape[1], 57)

        self.assertIsInstance(ds.training_sets["psal"], pl.DataFrame)
        self.assertEqual(ds.training_sets["psal"].shape[0], 126)
        self.assertEqual(ds.training_sets["psal"].shape[1], 57)

        self.assertIsInstance(ds.training_sets["pres"], pl.DataFrame)
        self.assertEqual(ds.training_sets["pres"].shape[0], 110)
        self.assertEqual(ds.training_sets["pres"].shape[1], 57)


class TestBuildModelClassLoader(unittest.TestCase):
    """
    Tests verifying that the correct build model class (BuildModel)
    is loaded from the config.
    """

    def setUp(self):
        """
        Initialize a training configuration and load input data for testing
        the build model class loader.
        """
        self.config_file_path = str(
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

        self.ds_input = load_step1_input_training_set(self.config)
        self.ds_input.input_file_names = self.input_file_names
        self.ds_input.process_targets()

    def test_load_dataset_valid_config(self):
        """
        Check that load_step4_build_model_class returns a BuildModel instance
        with the expected step name.
        """
        ds = load_step4_build_model_class(self.config)
        self.assertIsInstance(ds, BuildModel)
        self.assertEqual(ds.step_name, "build")

    def test_load_validation_suite_dataset(self):
        """
        Check that load_step2_model_validation_class returns a BuildModelSuite
        instance with the expected step name.
        """
        self.config.data["step_class_set"]["steps"]["model"] = "ModelSuite"
        self.config.data["step_class_set"]["steps"]["build"] = "BuildModelSuite"
        ds = load_step4_build_model_class(self.config)
        self.assertIsInstance(ds, BuildModelSuite)
        self.assertEqual(ds.step_name, "build")

    def test_training_and_test_sets(self):
        """
        Ensure that BuildModel receives the correct training and test sets
        when they are provided.
        """
        ds = load_step4_build_model_class(
            self.config, self.ds_input.training_sets, self.ds_input.test_sets
        )

        self.assertIsInstance(ds.training_sets["temp"], pl.DataFrame)
        self.assertEqual(ds.training_sets["temp"].shape[0], 116)
        self.assertEqual(ds.training_sets["temp"].shape[1], 57)

        self.assertIsInstance(ds.training_sets["psal"], pl.DataFrame)
        self.assertEqual(ds.training_sets["psal"].shape[0], 126)
        self.assertEqual(ds.training_sets["psal"].shape[1], 57)

        self.assertIsInstance(ds.training_sets["pres"], pl.DataFrame)
        self.assertEqual(ds.training_sets["pres"].shape[0], 110)
        self.assertEqual(ds.training_sets["pres"].shape[1], 57)

        self.assertIsInstance(ds.test_sets["temp"], pl.DataFrame)
        self.assertEqual(ds.test_sets["temp"].shape[0], 12)
        self.assertEqual(ds.test_sets["temp"].shape[1], 56)

        self.assertIsInstance(ds.test_sets["psal"], pl.DataFrame)
        self.assertEqual(ds.test_sets["psal"].shape[0], 14)
        self.assertEqual(ds.test_sets["psal"].shape[1], 56)

        self.assertIsInstance(ds.test_sets["pres"], pl.DataFrame)
        self.assertEqual(ds.test_sets["pres"].shape[0], 12)
        self.assertEqual(ds.test_sets["pres"].shape[1], 56)

    def test_suit_training_and_test_sets(self):
        """
        Ensure that BuildModel receives the correct training and test sets
        when they are provided.
        """
        self.config.data["step_class_set"]["steps"]["model"] = "ModelSuite"
        self.config.data["step_class_set"]["steps"]["build"] = "BuildModelSuite"
        ds = load_step4_build_model_class(
            self.config, self.ds_input.training_sets, self.ds_input.test_sets
        )

        self.assertIsInstance(ds.training_sets["temp"], pl.DataFrame)
        self.assertEqual(ds.training_sets["temp"].shape[0], 116)
        self.assertEqual(ds.training_sets["temp"].shape[1], 57)

        self.assertIsInstance(ds.training_sets["psal"], pl.DataFrame)
        self.assertEqual(ds.training_sets["psal"].shape[0], 126)
        self.assertEqual(ds.training_sets["psal"].shape[1], 57)

        self.assertIsInstance(ds.training_sets["pres"], pl.DataFrame)
        self.assertEqual(ds.training_sets["pres"].shape[0], 110)
        self.assertEqual(ds.training_sets["pres"].shape[1], 57)

        self.assertIsInstance(ds.test_sets["temp"], pl.DataFrame)
        self.assertEqual(ds.test_sets["temp"].shape[0], 12)
        self.assertEqual(ds.test_sets["temp"].shape[1], 56)

        self.assertIsInstance(ds.test_sets["psal"], pl.DataFrame)
        self.assertEqual(ds.test_sets["psal"].shape[0], 14)
        self.assertEqual(ds.test_sets["psal"].shape[1], 56)

        self.assertIsInstance(ds.test_sets["pres"], pl.DataFrame)
        self.assertEqual(ds.test_sets["pres"].shape[0], 12)
        self.assertEqual(ds.test_sets["pres"].shape[1], 56)
