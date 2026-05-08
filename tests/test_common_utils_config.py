"""
This module contains unit tests for the `read_config` function,
verifying its ability to correctly load configuration files
from various specified paths and handle error conditions.
"""

import unittest
from pathlib import Path

from aiqclib.common.utils.config import read_config


class TestReadConfig(unittest.TestCase):
    """
    A suite of tests verifying proper functionality of the read_config function
    under various usage scenarios (explicit file path, config name only,
    missing arguments, non-existent file).
    """

    def setUp(self):
        """
        Set up the test environment before each test method is run.
        Initializes the path to a sample configuration file for use in tests.
        """
        self.config_file_path = (
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_dataset_001.yaml"
        )

    def test_read_config_with_explicit_file(self):
        """
        Verify that read_config can load a specific YAML file when an explicit
        file path is provided. It checks for the presence of expected top-level
        keys ('data_sets', 'path_info_sets') in the loaded configuration.
        """
        config = read_config(config_file=str(self.config_file_path))
        self.assertIsNotNone(config, "Data should not be None")
        self.assertIn("data_sets", config, "Key 'data_sets' should be in the YAML")
        self.assertIn(
            "path_info_sets", config, "Key 'path_info_sets' should be in the YAML"
        )

    def test_read_config_no_params_raises_error(self):
        """
        Check that a TypeError is raised when `read_config` is called
        without providing either `config_file` or `config_file_name`,
        as at least one parameter is required for the function to proceed.
        """
        with self.assertRaises(TypeError):
            read_config()

    def test_read_config_nonexistent_file(self):
        """
        Ensure that a FileNotFoundError is raised when `read_config` is
        provided with an explicit `config_file` path that does not correspond
        to an existing file.
        """
        with self.assertRaises(FileNotFoundError):
            read_config(config_file="non_existent.yaml")
