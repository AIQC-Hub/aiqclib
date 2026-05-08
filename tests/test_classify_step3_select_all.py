"""Unit tests for the SelectDataSetAll class in aiqclib.classify.step3_select_profiles.

This module contains a suite of tests to verify the correct functionality
of the SelectDataSetAll class, including profile selection, labeling,
and persistence operations.
"""

import os
import unittest
from pathlib import Path

import polars as pl

from aiqclib.classify.step3_select_profiles.dataset_all import SelectDataSetAll
from aiqclib.common.config.classify_config import ClassificationConfig
from aiqclib.common.loader.classify_loader import load_classify_step1_input_dataset


class TestSelectDataSetA(unittest.TestCase):
    """
    A suite of tests ensuring the SelectDataSetAll class operates correctly
    for selecting and labeling profiles, as well as writing results to disk.
    """

    def setUp(self):
        """Set up test environment and load input dataset for all tests.

        This method initializes the configuration and loads the necessary
        input data for the tests, ensuring a consistent starting state.
        """
        self.config_file_path = str(
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_classify_001.yaml"
        )
        self.config = ClassificationConfig(str(self.config_file_path))
        self.config.select("NRT_BO_001")
        self.test_data_file = (
            Path(__file__).resolve().parent
            / "data"
            / "input"
            / "nrt_cora_bo_test.parquet"
        )
        self.ds = load_classify_step1_input_dataset(self.config)
        self.ds.input_file_name = str(self.test_data_file)
        self.ds.read_input_data()

    def test_step_name(self):
        """Ensure the step name is set correctly to 'select'.

        Verifies that the 'step_name' attribute of SelectDataSetAll
        is initialized to the expected value 'select'.
        """
        ds = SelectDataSetAll(self.config)
        self.assertEqual(ds.step_name, "select")

    def test_output_file_name(self):
        """Verify that the output file name is set based on configuration.

        Checks if the 'output_file_name' attribute correctly reflects
        the path derived from the provided classification configuration.
        """
        ds = SelectDataSetAll(self.config)
        self.assertEqual(
            "/path/to/select_1/nrt_bo_001/select_folder_1/selected_profiles_classify.parquet",
            str(ds.output_file_name),
        )

    def test_input_data(self):
        """Ensure input data is loaded into the class as a Polars DataFrame with expected shape.

        Confirms that the input data is correctly read and stored as a
        Polars DataFrame, and that its dimensions match the expected
        row and column counts.
        """
        ds = SelectDataSetAll(self.config, input_data=self.ds.input_data)
        self.assertIsInstance(ds.input_data, pl.DataFrame)
        self.assertEqual(ds.input_data.shape[0], 19480)
        self.assertEqual(ds.input_data.shape[1], 30)

    def test_selected_profiles(self):
        """Check that all profiles are selected correctly and the resulting DataFrame has the expected shape.

        Tests the 'select_all_profiles' method to ensure it populates
        'selected_profiles' with a Polars DataFrame of the anticipated
        size and column count after selection.
        """
        ds = SelectDataSetAll(self.config, input_data=self.ds.input_data)
        ds.select_all_profiles()
        self.assertIsInstance(ds.selected_profiles, pl.DataFrame)
        self.assertEqual(ds.selected_profiles.shape[0], 84)
        self.assertEqual(ds.selected_profiles.shape[1], 8)

    def test_label_profiles(self):
        """Check that profiles are labeled correctly, resulting in the expected DataFrame shape.

        Tests the 'label_profiles' method, confirming that 'selected_profiles'
        is populated with the correct number of rows and columns after labeling.
        """
        ds = SelectDataSetAll(self.config, input_data=self.ds.input_data)
        ds.label_profiles()
        self.assertEqual(ds.selected_profiles.shape[0], 84)
        self.assertEqual(ds.selected_profiles.shape[1], 8)

    def test_write_selected_profiles(self):
        """Confirm that selected profiles are written to a file successfully and the file exists.

        Verifies that the 'write_selected_profiles' method correctly
        persists the DataFrame to the specified output file path and
        that the file is created. The temporary file is then cleaned up.
        """
        ds = SelectDataSetAll(self.config, input_data=self.ds.input_data)
        ds.output_file_name = str(
            Path(__file__).resolve().parent
            / "data"
            / "select"
            / "temp_selected_profiles_classify.parquet"
        )

        # Assuming label_profiles populates selected_profiles before writing.
        # If write_selected_profiles can write without prior labeling,
        # this call might not be strictly necessary depending on the method's contract.
        ds.label_profiles()
        ds.write_selected_profiles()
        self.assertTrue(os.path.exists(ds.output_file_name))
        os.remove(ds.output_file_name)

    def test_write_empty_selected_profiles(self):
        """Check that writing an unpopulated (empty or None) selected profiles DataFrame raises a ValueError.

        Ensures that an attempt to write 'selected_profiles' when it has
        not been populated (or is empty/None) correctly raises a ValueError,
        preventing write operations on invalid data.
        """
        ds = SelectDataSetAll(self.config, input_data=self.ds.input_data)
        ds.output_file_name = str(
            Path(__file__).resolve().parent
            / "data"
            / "select"
            / "temp_selected_profiles_classify.parquet"
        )

        # selected_profiles is not populated by select_all_profiles() or label_profiles() here,
        # so it is expected to be empty or None, causing a ValueError upon writing.
        with self.assertRaises(ValueError):
            ds.write_selected_profiles()
