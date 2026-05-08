"""
This module contains unit tests for the ClassificationConfig class,
ensuring its ability to load, validate, and select configuration settings
for classification tasks, as well as generate correct file paths.
"""

import unittest
import pytest
from pathlib import Path

from aiqclib.common.config.classify_config import ClassificationConfig


class TestClassificationConfig(unittest.TestCase):
    """
    A suite of tests ensuring ClassificationConfig can validate configurations,
    select datasets correctly, and generate file/folder paths as expected.
    """

    def setUp(self):
        """
        Set up references to valid and template configuration files
        to be used in subsequent tests.
        """
        self.config_file_path = (
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_classify_001.yaml"
        )

    def test_valid_config(self):
        """
        Verify that validating a well-formed configuration reports it as valid.
        """
        ds = ClassificationConfig(str(self.config_file_path))
        msg = ds.validate()
        self.assertIn("valid", msg)

    def test_invalid_config(self):
        """
        Verify that validating a malformed configuration reports it as invalid.
        """
        config_file_path = (
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_dataset_invalid.yaml"
        )
        ds = ClassificationConfig(str(config_file_path))
        msg = ds.validate()
        self.assertIn("invalid", msg)

    def test_load_dataset_config(self):
        """
        Check that the expected configuration sections and their lengths
        are loaded from a valid configuration file.
        """
        ds = ClassificationConfig(str(self.config_file_path))
        ds.select("NRT_BO_001")

        self.assertEqual(len(ds.data["path_info"]), 8)
        self.assertEqual(len(ds.data["target_set"]), 2)
        self.assertEqual(len(ds.data["feature_set"]), 2)
        self.assertEqual(len(ds.data["feature_param_set"]), 2)
        self.assertEqual(len(ds.data["step_class_set"]), 2)
        self.assertEqual(len(ds.data["step_param_set"]), 2)

    def test_load_dataset_config_twice(self):
        """
        Confirm that calling `select()` multiple times with the same dataset
        name does not cause issues or alter the state incorrectly.
        """
        ds = ClassificationConfig(str(self.config_file_path))
        ds.select("NRT_BO_001")
        ds.select("NRT_BO_001")

    def test_invalid_dataset_name(self):
        """
        Check that attempting to select an unavailable dataset name
        raises a ValueError as expected.
        """
        ds = ClassificationConfig(str(self.config_file_path))
        with self.assertRaises(ValueError):
            ds.select("INVALID_NAME")


class TestClassificationConfigTemplate:
    @pytest.fixture(autouse=True)
    def setup_template(self):
        dir_config = Path(__file__).resolve().parent / "data" / "config"
        self.template_files = [
            dir_config / "config_classify_set_full_template.yaml",
            dir_config / "config_classify_set_template.yaml",
        ]

    @pytest.mark.parametrize("idx", range(2))
    def test_input_folder(self, idx):
        """
        Verify that full file paths for input files are generated correctly
        based on the configuration settings.
        """
        ds = ClassificationConfig(str(self.template_files[idx]))
        ds.select("classification_0001")
        input_file_name = ds.get_full_file_name(
            "input",
            ds.data["input_file_name"],
            use_dataset_folder=False,
            folder_name_auto=False,
        )
        assert input_file_name == "/path/to/input/nrt_cora_bo_4.parquet"

    @pytest.mark.parametrize("idx", range(2))
    def test_summary_folder(self, idx):
        """
        Confirm that full file paths for 'summary' folder items are resolved correctly.
        """
        ds = ClassificationConfig(str(self.template_files[idx]))
        ds.select("classification_0001")
        input_file_name = ds.get_full_file_name("summary", "test.txt")
        assert input_file_name == "/path/to/data/dataset_0001/summary/test.txt"

    @pytest.mark.parametrize("idx", range(2))
    def test_classify_folder(self, idx):
        """
        Confirm that full file paths for 'classify' folder items are resolved correctly.
        """
        ds = ClassificationConfig(str(self.template_files[idx]))
        ds.select("classification_0001")
        input_file_name = ds.get_full_file_name("classify", "test.txt")
        assert input_file_name == "/path/to/data/dataset_0001/classify/test.txt"

    @pytest.mark.parametrize("idx", range(2))
    def test_auto_select(self, idx):
        """
        Confirm that the `auto_select` option in the constructor correctly
        determines whether data is automatically loaded or not.
        """
        ds = ClassificationConfig(str(self.template_files[idx]), False)
        assert ds.data is None

        ds = ClassificationConfig(str(self.template_files[idx]), True)
        assert ds.data is not None
