"""
This module contains unit tests for the TrainingConfig class,
verifying its ability to validate training configurations,
handle dataset selection, and generate correct file/folder paths.
"""

import unittest
from pathlib import Path

from aiqclib.common.config.training_config import TrainingConfig


class TestTrainingConfig(unittest.TestCase):
    """
    A suite of tests verifying that TrainingConfig can validate a training configuration,
    handle dataset selection, and generate file/folder paths as expected.
    """

    def setUp(self):
        """
        Set up references to a valid training configuration file and
        a template configuration file to be used in subsequent tests.
        """
        self.config_file_path = (
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_training_001.yaml"
        )
        self.template_file = (
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "config_train_set_template.yaml"
        )

    def test_valid_config(self):
        """
        Confirm that a well-formed configuration is identified as valid
        by the validate() method.
        """
        ds = TrainingConfig(str(self.config_file_path))
        msg = ds.validate()
        self.assertIn("valid", msg)

    def test_invalid_config(self):
        """
        Confirm that an improperly formed configuration is identified as invalid.
        """
        config_file_path = (
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_dataset_invalid.yaml"
        )
        ds = TrainingConfig(str(config_file_path))
        msg = ds.validate()
        self.assertIn("invalid", msg)

    def test_load_dataset_config(self):
        """
        Check that the correct sections (path_info, target_set, step_class_set,
        step_param_set) are loaded from a valid training configuration.
        """
        ds = TrainingConfig(str(self.config_file_path))
        ds.select("NRT_BO_001")

        self.assertEqual(len(ds.data["path_info"]), 6)
        self.assertEqual(len(ds.data["target_set"]), 2)
        self.assertEqual(len(ds.data["step_class_set"]), 2)
        self.assertEqual(len(ds.data["step_param_set"]), 2)

    def test_invalid_dataset_name(self):
        """
        Ensure that selecting a non-existent dataset name triggers a ValueError.
        """
        ds = TrainingConfig(str(self.config_file_path))
        with self.assertRaises(ValueError):
            ds.select("INVALID_NAME")

    def test_input_folder(self):
        """
        Confirm that file paths for the 'input' folder are generated correctly
        when using the template configuration.
        """
        ds = TrainingConfig(str(self.template_file))
        ds.select("training_0001")
        input_file_name = ds.get_full_file_name("input", "test.txt")
        self.assertEqual(
            input_file_name, "/path/to/data/dataset_0001/training/test.txt"
        )

    def test_valid_folder(self):
        """
        Confirm that file paths for the 'valid' folder are generated correctly
        when using the template configuration.
        """
        ds = TrainingConfig(str(self.template_file))
        ds.select("training_0001")
        input_file_name = ds.get_full_file_name("valid", "test.txt")
        self.assertEqual(input_file_name, "/path/to/data/dataset_0001/valid/test.txt")

    def test_build_folder(self):
        """
        Confirm that file paths for the 'build' folder are generated correctly
        when using the template configuration.
        """
        ds = TrainingConfig(str(self.template_file))
        ds.select("training_0001")
        input_file_name = ds.get_full_file_name("build", "test.txt")
        self.assertEqual(input_file_name, "/path/to/data/dataset_0001/build/test.txt")

    def test_auto_select(self):
        """
        Confirm that auto select options works as expected, loading data
        when auto_select is True and not loading it when False.
        """
        ds = TrainingConfig(str(self.template_file), False)
        self.assertIsNone(ds.data)

        ds = TrainingConfig(str(self.template_file), True)
        self.assertIsNotNone(ds.data)
