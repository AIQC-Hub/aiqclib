"""Tests for the DataSetConfig class, verifying its configuration validation,
dataset selection, and path generation capabilities."""

import unittest
import pytest
from pathlib import Path

from aiqclib.common.config.dataset_config import DataSetConfig


class TestDataSetConfig(unittest.TestCase):
    """
    A suite of tests ensuring DataSetConfig can validate configurations,
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
            / "test_dataset_001.yaml"
        )

    def test_valid_config(self):
        """
        Verify that validating a well-formed configuration reports it as valid.
        """
        ds = DataSetConfig(str(self.config_file_path))
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
        ds = DataSetConfig(str(config_file_path))
        msg = ds.validate()
        self.assertIn("invalid", msg)

    def test_load_dataset_config(self):
        """
        Check that the correct sections (path_info, target_set, etc.)
        are loaded from a valid configuration.
        """
        ds = DataSetConfig(str(self.config_file_path))
        ds.select("NRT_BO_001")

        self.assertEqual(len(ds.data["path_info"]), 6)
        self.assertEqual(len(ds.data["target_set"]), 2)
        self.assertEqual(len(ds.data["feature_set"]), 2)
        self.assertEqual(len(ds.data["feature_param_set"]), 2)
        self.assertEqual(len(ds.data["step_class_set"]), 2)
        self.assertEqual(len(ds.data["step_param_set"]), 2)

    def test_load_dataset_config_twice(self):
        """
        Confirm that calling select() multiple times does not break anything.
        """
        ds = DataSetConfig(str(self.config_file_path))
        ds.select("NRT_BO_001")
        ds.select("NRT_BO_001")

    def test_invalid_dataset_name(self):
        """
        Check that attempting to select an unavailable dataset name
        raises a ValueError.
        """
        ds = DataSetConfig(str(self.config_file_path))
        with self.assertRaises(ValueError):
            ds.select("INVALID_NAME")


class TestDataSetConfigTemplate:
    @pytest.fixture(autouse=True)
    def setup_template(self):
        dir_config = Path(__file__).resolve().parent / "data" / "config"
        self.template_files = [
            dir_config / "config_data_set_full_template.yaml",
            dir_config / "config_data_set_reduced_template.yaml",
            dir_config / "config_data_set_template.yaml",
        ]

    @pytest.mark.parametrize("idx", range(2))
    def test_input_folder(self, idx):
        """
        Verify that input folder paths are generated as expected using template config.
        """
        ds = DataSetConfig(str(self.template_files[idx]))
        ds.select("dataset_0001")
        input_file_name = ds.get_full_file_name(
            "input",
            ds.data["input_file_name"],
            use_dataset_folder=False,
            folder_name_auto=False,
        )
        assert input_file_name == "/path/to/input/nrt_cora_bo_4.parquet"

    @pytest.mark.parametrize("idx", range(3))
    def test_summary_folder(self, idx):
        """
        Confirm that files placed in a 'summary' folder are resolved correctly.
        """
        ds = DataSetConfig(str(self.template_files[idx]))
        ds.select("dataset_0001")
        input_file_name = ds.get_full_file_name("summary", "test.txt")
        assert input_file_name == "/path/to/data/dataset_0001/summary/test.txt"

    @pytest.mark.parametrize("idx", range(3))
    def test_split_folder(self, idx):
        """
        Confirm that files placed in a 'split' folder are resolved correctly.
        """
        ds = DataSetConfig(str(self.template_files[idx]))
        ds.select("dataset_0001")
        input_file_name = ds.get_full_file_name("split", "test.txt")
        assert input_file_name == "/path/to/data/dataset_0001/training/test.txt"

    @pytest.mark.parametrize("idx", range(3))
    def test_auto_select(self, idx):
        """
        Confirm that auto select options works.
        """
        ds = DataSetConfig(str(self.template_files[idx]), False)
        assert ds.data is None

        ds = DataSetConfig(str(self.template_files[idx]), True)
        assert ds.data is not None
