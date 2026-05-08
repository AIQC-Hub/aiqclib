"""
This module contains unit tests for the SelectDataSetA class,
which is responsible for selecting, labeling, and managing profiles
within a dataset based on specific criteria.
"""

import os
import unittest
from pathlib import Path

import polars as pl

from aiqclib.common.config.dataset_config import DataSetConfig
from aiqclib.common.loader.dataset_loader import load_step1_input_dataset
from aiqclib.prepare.step3_select_profiles.dataset_all import SelectDataSetAll


class TestSelectDataSetAll(unittest.TestCase):
    """
    A suite of tests ensuring the SelectDataSetA class operates correctly
    for selecting and labeling profiles, as well as writing results to disk.
    """

    def setUp(self):
        """Set up test environment and load input dataset."""
        self.config_file_path = str(
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_dataset_005.yaml"
        )
        self.config = DataSetConfig(str(self.config_file_path))
        self.config.select("NRT_BO_001")
        self.test_data_file = (
            Path(__file__).resolve().parent
            / "data"
            / "input"
            / "nrt_cora_bo_test.parquet"
        )
        self.ds = load_step1_input_dataset(self.config)
        self.ds.input_file_name = str(self.test_data_file)
        self.ds.read_input_data()

    def test_step_name(self):
        """Ensure the step name is set correctly to 'select'."""
        ds = SelectDataSetAll(self.config)
        self.assertEqual(ds.step_name, "select")

    def test_input_data(self):
        """Ensure input data is loaded into the class as a Polars DataFrame and has expected dimensions."""
        ds = SelectDataSetAll(self.config, input_data=self.ds.input_data)
        self.assertIsInstance(ds.input_data, pl.DataFrame)
        self.assertEqual(ds.input_data.shape[0], 132342)
        self.assertEqual(ds.input_data.shape[1], 30)

    def test_positive_profiles(self):
        """Check that positive profiles are selected correctly based on criteria."""
        ds = SelectDataSetAll(self.config, input_data=self.ds.input_data)
        ds.select_positive_profiles()
        self.assertIsInstance(ds.pos_profile_df, pl.DataFrame)
        self.assertEqual(ds.pos_profile_df.shape[0], 25)
        self.assertEqual(ds.pos_profile_df.shape[1], 8)

    def test_negative_profiles(self):
        """Check that negative profiles are selected correctly after positive profiles."""
        ds = SelectDataSetAll(self.config, input_data=self.ds.input_data)
        ds.select_positive_profiles()
        ds.select_negative_profiles()
        self.assertIsInstance(ds.neg_profile_df, pl.DataFrame)
        self.assertEqual(ds.neg_profile_df.shape[0], 478)
        self.assertEqual(ds.neg_profile_df.shape[1], 8)

    def test_label_profiles(self):
        """Check that profiles are labeled correctly and combined into a single DataFrame."""
        ds = SelectDataSetAll(self.config, input_data=self.ds.input_data)
        ds.label_profiles()
        self.assertEqual(ds.selected_profiles.shape[0], 503)
        self.assertEqual(ds.selected_profiles.shape[1], 8)

    def test_write_selected_profiles(self):
        """Confirm that selected profiles are written to a file successfully and the file exists."""
        ds = SelectDataSetAll(self.config, input_data=self.ds.input_data)
        ds.output_file_name = str(
            Path(__file__).resolve().parent
            / "data"
            / "select"
            / "temp_selected_profiles_all.parquet"
        )

        ds.label_profiles()
        ds.write_selected_profiles()
        self.assertTrue(os.path.exists(ds.output_file_name))
        os.remove(ds.output_file_name)

    def test_write_empty_selected_profiles(self):
        """Check that writing empty profiles (i.e., before labeling) raises a ValueError."""
        ds = SelectDataSetAll(self.config, input_data=self.ds.input_data)
        ds.output_file_name = str(
            Path(__file__).resolve().parent / "data" / "select"
            "temp_selected_profiles_all.parquet"
        )

        with self.assertRaises(ValueError):
            ds.write_selected_profiles()
