"""
This module contains unit tests for the InputDataSetAll class, which is responsible
for reading and resolving input data files within the aiqclib classification library.
The tests ensure that data loading from various file types (e.g., Parquet) works
correctly, file paths are resolved as expected, and reading options are applied.
"""

import unittest
from pathlib import Path

import polars as pl

from aiqclib.classify.step1_read_input.dataset_all import InputDataSetAll
from aiqclib.common.config.classify_config import ClassificationConfig


class TestInputDataSetAll(unittest.TestCase):
    """
    Tests for verifying input data reading and resolution in the InputDataSetAll class.
    Ensures data is loaded as expected from Parquet files, file names are resolved
    properly, and property checks are correct.
    """

    def setUp(self):
        """
        Set up the test configuration objects and specify test data file paths.
        This method is called before each test function to ensure a clean state.
        """
        self.config_file_path = (
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_classify_001.yaml"
        )
        self.config = ClassificationConfig(str(self.config_file_path))
        # Select a specific configuration profile for the tests
        self.config.select("NRT_BO_001")
        self.test_data_file = (
            Path(__file__).resolve().parent
            / "data"
            / "input"
            / "nrt_cora_bo_test.parquet"
        )

    def _get_input_data(self, config, file_type=None, read_file_options=None):
        """
        Helper method that loads input data into a Polars DataFrame, optionally
        setting the file type and read options in the config before reading.
        This method is used by multiple tests to abstract the data loading process.
        """
        ds = InputDataSetAll(config)
        # Manually set the input file name to a known test data file for consistent reading tests.
        ds.input_file_name = str(self.test_data_file)

        if file_type is not None:
            # Dynamically set the file type in the configuration for testing specific file formats.
            ds.config.data["step_param_set"]["steps"]["input"]["file_type"] = file_type

        if read_file_options is not None:
            # Dynamically set read file options in the configuration for testing data loading parameters.
            ds.config.data["step_param_set"]["steps"]["input"]["read_file_options"] = (
                read_file_options
            )

        ds.read_input_data()
        return ds.input_data

    def test_step_name(self):
        """
        Confirm that InputDataSetAll instances correctly declare their step name as "input".
        This verifies a fundamental property of the class for pipeline integration.
        """
        ds = InputDataSetAll(self.config)
        self.assertEqual(ds.step_name, "input")

    def test_input_file_name(self):
        """
        Ensure the input file name property reflects the exact path configured in the YAML file.
        This tests that the configured input file path string is correctly loaded into the class instance.
        """
        ds = InputDataSetAll(self.config)
        self.assertEqual(
            "/path/to/input_1/input_folder_1/nrt_cora_bo_test.parquet",
            str(ds.input_file_name),
        )

    def test_read_input_data_with_explicit_type(self):
        """
        Verify that data is successfully read from a Parquet file when 'parquet' is
        explicitly specified as the file_type in the configuration.
        """
        df = self._get_input_data(
            self.config, file_type="parquet", read_file_options={}
        )
        self.assertIsInstance(df, pl.DataFrame)
        self.assertEqual(df.shape[0], 19480)
        self.assertEqual(df.shape[1], 30)

    def test_read_input_data_infer_type(self):
        """
        Verify that data reading automatically infers the file type (from the file extension)
        when 'file_type' is not explicitly set in the configuration.
        """
        df = self._get_input_data(self.config, file_type=None, read_file_options={})
        self.assertIsInstance(df, pl.DataFrame)
        self.assertEqual(df.shape[0], 19480)
        self.assertEqual(df.shape[1], 30)

    def test_read_input_data_missing_options(self):
        """
        Confirm that data can be read successfully when no additional
        file reading options are provided in the configuration.
        """
        df = self._get_input_data(
            self.config, file_type="parquet", read_file_options=None
        )
        self.assertIsInstance(df, pl.DataFrame)
        self.assertEqual(df.shape[0], 19480)
        self.assertEqual(df.shape[1], 30)

    def test_read_input_data_file_not_found(self):
        """
        Ensure that attempting to read a non-existent input file correctly
        raises a FileNotFoundError, indicating robust error handling.
        """
        ds = InputDataSetAll(self.config)
        # Modify the input file name to point to a non-existent file path
        ds.input_file_name = str(self.test_data_file) + "_not_found"

        with self.assertRaises(FileNotFoundError):
            ds.read_input_data()

    def test_read_input_data_with_extra_options(self):
        """
        Verify that additional reading options (e.g., 'n_rows' to limit rows)
        can be passed via the configuration and are correctly applied during data loading.
        """
        df = self._get_input_data(
            self.config, file_type="parquet", read_file_options={"n_rows": 100}
        )
        self.assertIsInstance(df, pl.DataFrame)
        self.assertEqual(df.shape[0], 100)
        self.assertEqual(df.shape[1], 30)
