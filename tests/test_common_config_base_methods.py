"""Unit tests for the DataSetConfig class.

This module verifies the correct functionality of DataSetConfig's methods
related to path handling, base class retrieval, target variable management,
and summary statistics retrieval from configuration files.
"""

import unittest
from pathlib import Path

from aiqclib.common.config.dataset_config import DataSetConfig


class TestBaseConfigPathMethods(unittest.TestCase):
    """A suite of tests that verify the correctness of path-related methods
    within the DataSetConfig (e.g., base paths, file names, folder names).
    """

    def setUp(self):
        """Set up a reference to the test configuration file (test_dataset_001.yaml).

        This file is used by all subsequent tests in this class to
        initialize the `DataSetConfig` instance.
        """
        self.config_file_path = (
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_dataset_001.yaml"
        )

    def test_common_base_path(self):
        """Confirm that the 'common' base path is retrieved correctly from the config."""
        ds = DataSetConfig(str(self.config_file_path))
        ds.select("NRT_BO_001")

        base_path = ds.get_base_path("common")
        self.assertEqual("/path/to/data_1", base_path)

    def test_input_base_path(self):
        """Confirm that the 'input' base path is retrieved correctly when present."""
        ds = DataSetConfig(str(self.config_file_path))
        ds.select("NRT_BO_001")

        base_path = ds.get_base_path("input")
        self.assertEqual("/path/to/input_1", base_path)

    def test_default_base_path(self):
        """Ensure that if a specific base path is not found for a step, the common one is returned."""
        config_file_path = (
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_dataset_002.yaml"
        )
        ds = DataSetConfig(str(config_file_path))
        ds.select("NRT_BO_001")

        base_path = ds.get_base_path("locate")
        self.assertEqual("/path/to/data_1", base_path)

    def test_input_step_folder_name(self):
        """Verify that the folder name for the 'input' step is returned correctly."""
        ds = DataSetConfig(str(self.config_file_path))
        ds.select("NRT_BO_001")

        folder_name = ds.get_step_folder_name("input")
        self.assertEqual("input_folder_1", folder_name)

    def test_auto_select_step_folder_name(self):
        """Verify that the folder name is auto-selected when not explicitly defined."""
        config_file_path = (
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_dataset_002.yaml"
        )
        ds = DataSetConfig(str(config_file_path))
        ds.select("NRT_BO_001")

        folder_name = ds.get_step_folder_name("select")
        self.assertEqual("select", folder_name)

    def test_no_auto_select_step_folder_name(self):
        """Confirm that folder_name_auto=False returns an empty string
        if no folder name is defined in the config for the step.
        """
        config_file_path = (
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_dataset_002.yaml"
        )
        ds = DataSetConfig(str(config_file_path))
        ds.select("NRT_BO_001")

        folder_name = ds.get_step_folder_name("select", folder_name_auto=False)
        self.assertEqual("", folder_name)

    def test_common_dataset_folder_name(self):
        """Verify that the correct dataset folder name is retrieved for the 'input' step."""
        ds = DataSetConfig(str(self.config_file_path))
        ds.select("NRT_BO_001")

        dataset_folder_name = ds.get_dataset_folder_name("input")
        self.assertEqual("nrt_bo_001", dataset_folder_name)

    def test_dataset_folder_name_in_step_params(self):
        """Verify that an overridden dataset folder name is retrieved from step parameters."""
        config_file_path = (
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_dataset_002.yaml"
        )
        ds = DataSetConfig(str(config_file_path))
        ds.select("NRT_BO_001")

        dataset_folder_name = ds.get_dataset_folder_name("summary")
        self.assertEqual("summary_dataset_folder", dataset_folder_name)

    def test_default_file_name(self):
        """Verify that if a file name is missing in the config, the provided default is used."""
        ds = DataSetConfig(str(self.config_file_path))
        ds.select("NRT_BO_001")

        file_name = ds.get_file_name("input", "default_file.txt")
        self.assertEqual("default_file.txt", file_name)

    def test_no_default_file_name(self):
        """Check that an error is raised if no default is provided and the file name is missing."""
        ds = DataSetConfig(str(self.config_file_path))
        ds.select("NRT_BO_001")

        with self.assertRaises(ValueError):
            _ = ds.get_file_name("input")

    def test_file_name_in_params(self):
        """Ensure that if a file name is specified in step parameters, it is retrieved successfully."""
        config_file_path = (
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_dataset_002.yaml"
        )
        ds = DataSetConfig(str(config_file_path))
        ds.select("NRT_BO_001")

        file_name = ds.get_file_name("summary")
        self.assertEqual("summary_in_params.txt", file_name)

    def test_full_input_path(self):
        """Confirm that full file names are constructed properly when dataset folder usage is disabled."""
        ds = DataSetConfig(str(self.config_file_path))
        ds.select("NRT_BO_001")

        full_file_name = ds.get_full_file_name(
            "input", "test_input_file.txt", use_dataset_folder=False
        )
        self.assertEqual(
            full_file_name, "/path/to/input_1/input_folder_1/test_input_file.txt"
        )

    def test_full_input_path_with_dataset_folder(self):
        """Confirm that full file names are constructed properly when dataset folder usage is enabled."""
        ds = DataSetConfig(str(self.config_file_path))
        ds.select("NRT_BO_001")

        full_file_name = ds.get_full_file_name("input", "test_input_file.txt")
        self.assertEqual(
            full_file_name,
            "/path/to/input_1/nrt_bo_001/input_folder_1/test_input_file.txt",
        )

    def test_full_summary_path(self):
        """Check that a full file name is constructed using parameters from test_dataset_002.yaml."""
        config_file_path = (
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_dataset_002.yaml"
        )
        ds = DataSetConfig(str(config_file_path))
        ds.select("NRT_BO_001")

        full_file_name = ds.get_full_file_name("summary", "test_input_file.txt")
        self.assertEqual(
            full_file_name,
            "/path/to/data_1/summary_dataset_folder/summary/summary_in_params.txt",
        )


class TestBaseConfigBaseClass(unittest.TestCase):
    """A suite of tests that verify lookup of base classes for steps
    within the DataSetConfig.
    """

    def setUp(self):
        """Set up a reference to the test configuration file (test_dataset_001.yaml).

        This file is used by all subsequent tests in this class.
        """
        self.config_file_path = (
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_dataset_001.yaml"
        )

    def test_input_base_class(self):
        """Check that the correct base class name is returned for the 'input' step."""
        ds = DataSetConfig(str(self.config_file_path))
        ds.select("NRT_BO_001")

        base_class = ds.get_base_class("input")
        self.assertEqual("InputDataSetA", base_class)


class TestBaseConfigTargets(unittest.TestCase):
    """Tests focusing on how target variables, names, and related file names
    are retrieved from DataSetConfig.
    """

    def setUp(self):
        """Set up a reference to the test configuration file (test_dataset_001.yaml).

        This file is used by all subsequent tests in this class.
        """
        self.config_file_path = (
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_dataset_001.yaml"
        )

    def test_target_variables(self):
        """Confirm that `get_target_variables()` returns the expected number of targets."""
        ds = DataSetConfig(str(self.config_file_path))
        ds.select("NRT_BO_001")

        target_variables = ds.get_target_variables()
        self.assertEqual(len(target_variables), 3)

    def test_target_names(self):
        """Confirm that `get_target_names()` returns a list of correct target variable names."""
        ds = DataSetConfig(str(self.config_file_path))
        ds.select("NRT_BO_001")

        target_names = ds.get_target_names()
        self.assertEqual(target_names, ["temp", "psal", "pres"])

    def test_target_dict(self):
        """Ensure that `get_target_dict()` returns a dictionary detailing each target."""
        ds = DataSetConfig(str(self.config_file_path))
        ds.select("NRT_BO_001")

        target_dict = ds.get_target_dict()
        self.assertEqual(
            target_dict["temp"],
            {
                "name": "temp",
                "flag": "temp_qc",
                "pos_flag_values": [4],
                "neg_flag_values": [1],
            },
        )
        self.assertEqual(
            target_dict["psal"],
            {
                "name": "psal",
                "flag": "psal_qc",
                "pos_flag_values": [4],
                "neg_flag_values": [1],
            },
        )
        self.assertEqual(
            target_dict["pres"],
            {
                "name": "pres",
                "flag": "pres_qc",
                "pos_flag_values": [4],
                "neg_flag_values": [1],
            },
        )

    def test_target_file_names(self):
        """Confirm that target file names are generated as expected
        using a placeholder format string.
        """
        ds = DataSetConfig(str(self.config_file_path))
        ds.select("NRT_BO_001")

        target_file_names = ds.get_target_file_names(
            "select", "{target_name}_features.parquet"
        )
        self.assertEqual(
            target_file_names["temp"],
            "/path/to/select_1/nrt_bo_001/select_folder_1/temp_features.parquet",
        )
        self.assertEqual(
            target_file_names["psal"],
            "/path/to/select_1/nrt_bo_001/select_folder_1/psal_features.parquet",
        )
        self.assertEqual(
            target_file_names["pres"],
            "/path/to/select_1/nrt_bo_001/select_folder_1/pres_features.parquet",
        )


class TestBaseConfigSummaryStats(unittest.TestCase):
    """Tests focusing on how summary statistics are retrieved from DataSetConfig."""

    def setUp(self):
        """Set up a reference to the test configuration file (test_dataset_001.yaml).

        This file is used by all subsequent tests in this class.
        """
        self.config_file_path = (
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_dataset_001.yaml"
        )

    def test_config_location_summary_stats(self):
        """Confirm that location summary statistics are retrieved correctly."""
        ds = DataSetConfig(str(self.config_file_path))
        ds.select("NRT_BO_001")
        stats = ds.get_summary_stats("location")

        self.assertIn("longitude", stats)
        self.assertIn("latitude", stats)

    def test_config_profile_summary_stats(self):
        """Confirm that profile summary statistics for a specific set are retrieved correctly."""
        ds = DataSetConfig(str(self.config_file_path))
        ds.select("NRT_BO_001")
        stats = ds.get_summary_stats("profile_summary_stats")

        self.assertIn("pres", stats)
        self.assertIn("mean", stats["pres"])
        self.assertIn("median", stats["pres"])

    def test_config_basic_values3_stats(self):
        """Confirm that basic_values3 summary statistics are retrieved correctly."""
        ds = DataSetConfig(str(self.config_file_path))
        ds.select("NRT_BO_001")
        stats = ds.get_summary_stats("basic_values3")

        self.assertIn("pres", stats)
        self.assertIn("min", stats["pres"])
        self.assertEqual(0, stats["pres"]["min"])

    def test_update_feature_param_with_stats(self):
        """Verify that feature parameters are updated with relevant summary statistics."""
        ds = DataSetConfig(str(self.config_file_path))
        ds.select("NRT_BO_001")

        # Iterate through params to check if 'stats' key was added
        for x in ds.data["feature_param_set"]["params"]:
            if "stats_set" in x:
                self.assertIn("stats", x)
