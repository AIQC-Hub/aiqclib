"""
This module contains unit tests for the SplitDataSetA class, ensuring its
correct functionality in splitting extracted feature datasets into training
and test sets, generating appropriate output file paths, and adhering to
configuration parameters like test set fraction and k-fold validation.
It also verifies the integrity of the dataframes after splitting and
the successful writing of these sets to disk.
"""

import os
import unittest
from pathlib import Path

import polars as pl

from aiqclib.common.config.dataset_config import DataSetConfig
from aiqclib.common.loader.dataset_loader import (
    load_step1_input_dataset,
    load_step2_summary_dataset,
    load_step3_select_dataset,
    load_step4_locate_dataset,
    load_step5_extract_dataset,
)
from aiqclib.prepare.step6_split_dataset.dataset_all import SplitDataSetAll


class TestSplitDataSetAll(unittest.TestCase):
    """
    A suite of unit tests ensuring SplitDataSetA correctly splits extracted features
    into training and test sets, writes them to files, and respects user-defined
    configurations such as test set fraction and k-fold.
    """

    def setUp(self):
        """
        Set up test environment and load data from previous steps
        (input, summary, select, locate, extract) to provide necessary
        dependencies for SplitDataSetA.
        """
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
        self.ds_input = load_step1_input_dataset(self.config)
        self.ds_input.input_file_name = str(self.test_data_file)
        self.ds_input.read_input_data()

        self.ds_summary = load_step2_summary_dataset(
            self.config, input_data=self.ds_input.input_data
        )
        self.ds_summary.calculate_stats()

        self.ds_select = load_step3_select_dataset(
            self.config, input_data=self.ds_input.input_data
        )
        self.ds_select.label_profiles()

        self.ds_locate = load_step4_locate_dataset(
            self.config,
            input_data=self.ds_input.input_data,
            selected_profiles=self.ds_select.selected_profiles,
        )
        self.ds_locate.process_targets()

        self.ds_extract = load_step5_extract_dataset(
            self.config,
            input_data=self.ds_input.input_data,
            selected_profiles=self.ds_select.selected_profiles,
            selected_rows=self.ds_locate.selected_rows,
            summary_stats=self.ds_summary.summary_stats,
        )
        self.ds_extract.process_targets()

    def test_step_name(self):
        """
        Verify that the step name attribute of SplitDataSetA is correctly
        set to 'split'.
        """
        ds = SplitDataSetAll(self.config)
        self.assertEqual(ds.step_name, "split")

    def test_target_features_data(self):
        """
        Check that target features (extracted dataframes) are correctly
        loaded into the SplitDataSetA class upon initialization,
        verifying their type and dimensions.
        """
        ds = SplitDataSetAll(
            self.config, target_features=self.ds_extract.target_features
        )

        self.assertIsInstance(ds.target_features["temp"], pl.DataFrame)
        self.assertEqual(ds.target_features["temp"].shape[0], 132342)
        self.assertEqual(ds.target_features["temp"].shape[1], 58)

        self.assertIsInstance(ds.target_features["psal"], pl.DataFrame)
        self.assertEqual(ds.target_features["psal"].shape[0], 132342)
        self.assertEqual(ds.target_features["psal"].shape[1], 58)

        self.assertIsInstance(ds.target_features["pres"], pl.DataFrame)
        self.assertEqual(ds.target_features["pres"].shape[0], 132342)
        self.assertEqual(ds.target_features["pres"].shape[1], 58)

    def test_split_features_data(self):
        """
        Verify the splitting of features into training and test sets,
        checking the resulting dimensions of the dataframes for both
        "temp" and "psal" target variables.
        """
        ds = SplitDataSetAll(
            self.config, target_features=self.ds_extract.target_features
        )

        ds.process_targets()

        self.assertIsInstance(ds.training_sets["temp"], pl.DataFrame)
        self.assertEqual(ds.training_sets["temp"].shape[0], 119109)
        self.assertEqual(ds.training_sets["temp"].shape[1], 57)

        self.assertIsInstance(ds.test_sets["temp"], pl.DataFrame)
        self.assertEqual(ds.test_sets["temp"].shape[0], 13233)
        self.assertEqual(ds.test_sets["temp"].shape[1], 56)

        self.assertIsInstance(ds.training_sets["psal"], pl.DataFrame)
        self.assertEqual(ds.training_sets["psal"].shape[0], 119108)
        self.assertEqual(ds.training_sets["psal"].shape[1], 57)

        self.assertIsInstance(ds.test_sets["psal"], pl.DataFrame)
        self.assertEqual(ds.test_sets["psal"].shape[0], 13234)
        self.assertEqual(ds.test_sets["psal"].shape[1], 56)

        self.assertIsInstance(ds.training_sets["pres"], pl.DataFrame)
        self.assertEqual(ds.training_sets["pres"].shape[0], 119108)
        self.assertEqual(ds.training_sets["pres"].shape[1], 57)

        self.assertIsInstance(ds.test_sets["pres"], pl.DataFrame)
        self.assertEqual(ds.test_sets["pres"].shape[0], 13234)
        self.assertEqual(ds.test_sets["pres"].shape[1], 56)

    def test_write_training_sets(self):
        """
        Confirm that training sets for each target variable are
        successfully written to their respective parquet files,
        and clean up the created files afterwards.
        """
        ds = SplitDataSetAll(
            self.config, target_features=self.ds_extract.target_features
        )
        ds.process_targets()

        data_path = Path(__file__).resolve().parent / "data" / "training"
        # Ensure the directory exists
        data_path.mkdir(parents=True, exist_ok=True)

        ds.output_file_names["train"]["temp"] = str(
            data_path / "temp_train_set_all_temp.parquet"
        )
        ds.output_file_names["train"]["psal"] = str(
            data_path / "temp_train_set_all_psal.parquet"
        )
        ds.output_file_names["train"]["pres"] = str(
            data_path / "temp_train_set_all_pres.parquet"
        )

        ds.write_training_sets()

        self.assertTrue(os.path.exists(ds.output_file_names["train"]["temp"]))
        self.assertTrue(os.path.exists(ds.output_file_names["train"]["psal"]))
        self.assertTrue(os.path.exists(ds.output_file_names["train"]["pres"]))

        os.remove(ds.output_file_names["train"]["temp"])
        os.remove(ds.output_file_names["train"]["psal"])
        os.remove(ds.output_file_names["train"]["pres"])

    def test_write_test_sets(self):
        """
        Confirm that test sets for each target variable are
        successfully written to their respective parquet files,
        and clean up the created files afterwards.
        """
        ds = SplitDataSetAll(
            self.config, target_features=self.ds_extract.target_features
        )
        ds.process_targets()

        data_path = Path(__file__).resolve().parent / "data" / "training"
        # Ensure the directory exists
        data_path.mkdir(parents=True, exist_ok=True)

        ds.output_file_names["test"]["temp"] = str(
            data_path / "temp_test_set_all_temp.parquet"
        )
        ds.output_file_names["test"]["psal"] = str(
            data_path / "temp_test_set_all_psal.parquet"
        )
        ds.output_file_names["test"]["pres"] = str(
            data_path / "temp_test_set_all_pres.parquet"
        )

        ds.write_test_sets()

        self.assertTrue(os.path.exists(ds.output_file_names["test"]["temp"]))
        self.assertTrue(os.path.exists(ds.output_file_names["test"]["psal"]))
        self.assertTrue(os.path.exists(ds.output_file_names["test"]["pres"]))

        os.remove(ds.output_file_names["test"]["temp"])
        os.remove(ds.output_file_names["test"]["psal"])
        os.remove(ds.output_file_names["test"]["pres"])
