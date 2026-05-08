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
from aiqclib.prepare.step3_select_profiles.dataset_a import SelectDataSetA


class TestSelectDataSetA(unittest.TestCase):
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
            / "test_dataset_001.yaml"
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
        ds = SelectDataSetA(self.config)
        self.assertEqual(ds.step_name, "select")

    def test_output_file_name(self):
        """Verify that the output file name is set based on configuration."""
        ds = SelectDataSetA(self.config)
        self.assertEqual(
            "/path/to/select_1/nrt_bo_001/select_folder_1/selected_profiles.parquet",
            str(ds.output_file_name),
        )

    def test_default_output_file_name(self):
        """Check a default output file name from a different configuration."""
        config_file_path = str(
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_dataset_002.yaml"
        )
        config = DataSetConfig(config_file_path)
        config.select("NRT_BO_001")

        ds = SelectDataSetA(config)
        self.assertEqual(
            "/path/to/data_1/nrt_bo_001/select/selected_profiles.parquet",
            str(ds.output_file_name),
        )

    def test_input_data(self):
        """Ensure input data is loaded into the class as a Polars DataFrame and has expected dimensions."""
        ds = SelectDataSetA(self.config, input_data=self.ds.input_data)
        self.assertIsInstance(ds.input_data, pl.DataFrame)
        self.assertEqual(ds.input_data.shape[0], 132342)
        self.assertEqual(ds.input_data.shape[1], 30)

    def test_positive_profiles(self):
        """Check that positive profiles are selected correctly based on criteria."""
        ds = SelectDataSetA(self.config, input_data=self.ds.input_data)
        ds.select_positive_profiles()
        self.assertIsInstance(ds.pos_profile_df, pl.DataFrame)
        self.assertEqual(ds.pos_profile_df.shape[0], 25)
        self.assertEqual(ds.pos_profile_df.shape[1], 7)

    def test_negative_profiles(self):
        """Check that negative profiles are selected correctly after positive profiles."""
        ds = SelectDataSetA(self.config, input_data=self.ds.input_data)
        ds.select_positive_profiles()
        ds.select_negative_profiles()
        self.assertIsInstance(ds.neg_profile_df, pl.DataFrame)
        self.assertEqual(ds.neg_profile_df.shape[0], 478)
        self.assertEqual(ds.neg_profile_df.shape[1], 7)

    def test_find_profile_pairs(self):
        """Validate the creation of matching positive and negative profile pairs."""
        ds = SelectDataSetA(self.config, input_data=self.ds.input_data)
        ds.select_positive_profiles()
        ds.select_negative_profiles()
        ds.find_profile_pairs()
        self.assertEqual(ds.pos_profile_df.shape[0], 25)
        self.assertEqual(ds.pos_profile_df.shape[1], 8)
        self.assertEqual(ds.neg_profile_df.shape[0], 25)
        self.assertEqual(ds.neg_profile_df.shape[1], 8)

    def test_label_profiles(self):
        """Check that profiles are labeled correctly and combined into a single DataFrame."""
        ds = SelectDataSetA(self.config, input_data=self.ds.input_data)
        ds.label_profiles()
        self.assertEqual(ds.selected_profiles.shape[0], 50)
        self.assertEqual(ds.selected_profiles.shape[1], 8)

    def test_write_selected_profiles(self):
        """Confirm that selected profiles are written to a file successfully and the file exists."""
        ds = SelectDataSetA(self.config, input_data=self.ds.input_data)
        ds.output_file_name = str(
            Path(__file__).resolve().parent
            / "data"
            / "select"
            / "temp_selected_profiles.parquet"
        )

        ds.label_profiles()
        ds.write_selected_profiles()
        self.assertTrue(os.path.exists(ds.output_file_name))
        os.remove(ds.output_file_name)

    def test_write_empty_selected_profiles(self):
        """Check that writing empty profiles (i.e., before labeling) raises a ValueError."""
        ds = SelectDataSetA(self.config, input_data=self.ds.input_data)
        ds.output_file_name = str(
            Path(__file__).resolve().parent / "data" / "select"
            "temp_selected_profiles.parquet"
        )

        with self.assertRaises(ValueError):
            ds.write_selected_profiles()


class TestSelectDataSetANegX5(unittest.TestCase):
    """
    A suite of tests ensuring the SelectDataSetA class operates correctly
    for selecting and labeling profiles, as well as writing results to disk,
    specifically when a different negative to positive ratio is configured.
    """

    def setUp(self):
        """Set up test environment with a configuration specifying a neg_pos_ratio of 5."""
        self.config_file_path = str(
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_dataset_003.yaml"
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

    def test_neg_pos_ratio(self):
        """Verify that the configured negative to positive ratio is correctly loaded."""
        ds = SelectDataSetA(self.config, input_data=self.ds.input_data)
        self.assertEqual(ds.config.get_step_params("select").get("neg_pos_ratio", 1), 5)

    def test_find_profile_pairs(self):
        """Validate the creation of matching profile pairs with a 1:5 positive:negative ratio."""
        ds = SelectDataSetA(self.config, input_data=self.ds.input_data)
        ds.select_positive_profiles()
        ds.select_negative_profiles()
        ds.find_profile_pairs()
        self.assertEqual(ds.pos_profile_df.shape[0], 25)
        self.assertEqual(ds.pos_profile_df.shape[1], 8)
        self.assertEqual(ds.neg_profile_df.shape[0], 125)  # 25 positive * 5 ratio
        self.assertEqual(ds.neg_profile_df.shape[1], 8)

    def test_label_profiles(self):
        """Check that profiles are labeled correctly and combined, reflecting the 1:5 ratio."""
        ds = SelectDataSetA(self.config, input_data=self.ds.input_data)
        ds.label_profiles()
        self.assertEqual(
            ds.selected_profiles.shape[0], 150
        )  # 25 positive + 125 negative
        self.assertEqual(ds.selected_profiles.shape[1], 8)

    def test_write_selected_profiles(self):
        """Confirm that selected profiles (with 1:5 ratio) are written to a file successfully."""
        ds = SelectDataSetA(self.config, input_data=self.ds.input_data)
        ds.output_file_name = str(
            Path(__file__).resolve().parent
            / "data"
            / "select"
            / "temp_selected_profiles.parquet"
        )

        ds.label_profiles()
        ds.write_selected_profiles()
        self.assertTrue(os.path.exists(ds.output_file_name))
        os.remove(ds.output_file_name)
