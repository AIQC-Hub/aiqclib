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
from aiqclib.prepare.step6_split_dataset.dataset_a import SplitDataSetA


class TestSplitDataSetA(unittest.TestCase):
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
        ds = SplitDataSetA(self.config)
        self.assertEqual(ds.step_name, "split")

    def test_output_file_names(self):
        """
        Ensure the default output file names for training and test sets
        are correctly generated based on the configuration for each
        target variable.
        """
        ds = SplitDataSetA(self.config)

        self.assertEqual(
            "/path/to/split_1/nrt_bo_001/split_folder_1/train_set_temp.parquet",
            str(ds.output_file_names["train"]["temp"]),
        )
        self.assertEqual(
            "/path/to/split_1/nrt_bo_001/split_folder_1/train_set_psal.parquet",
            str(ds.output_file_names["train"]["psal"]),
        )
        self.assertEqual(
            "/path/to/split_1/nrt_bo_001/split_folder_1/train_set_pres.parquet",
            str(ds.output_file_names["train"]["pres"]),
        )
        self.assertEqual(
            "/path/to/split_1/nrt_bo_001/split_folder_1/test_set_temp.parquet",
            str(ds.output_file_names["test"]["temp"]),
        )
        self.assertEqual(
            "/path/to/split_1/nrt_bo_001/split_folder_1/test_set_psal.parquet",
            str(ds.output_file_names["test"]["psal"]),
        )
        self.assertEqual(
            "/path/to/split_1/nrt_bo_001/split_folder_1/test_set_pres.parquet",
            str(ds.output_file_names["test"]["pres"]),
        )

    def test_target_features_data(self):
        """
        Check that target features (extracted dataframes) are correctly
        loaded into the SplitDataSetA class upon initialization,
        verifying their type and dimensions.
        """
        ds = SplitDataSetA(self.config, target_features=self.ds_extract.target_features)

        self.assertIsInstance(ds.target_features["temp"], pl.DataFrame)
        self.assertEqual(ds.target_features["temp"].shape[0], 128)
        self.assertEqual(ds.target_features["temp"].shape[1], 58)

        self.assertIsInstance(ds.target_features["psal"], pl.DataFrame)
        self.assertEqual(ds.target_features["psal"].shape[0], 140)
        self.assertEqual(ds.target_features["psal"].shape[1], 58)

        self.assertIsInstance(ds.target_features["pres"], pl.DataFrame)
        self.assertEqual(ds.target_features["pres"].shape[0], 122)
        self.assertEqual(ds.target_features["pres"].shape[1], 58)

    def test_split_features_data(self):
        """
        Verify the splitting of features into training and test sets,
        checking the resulting dimensions of the dataframes for both
        "temp" and "psal" target variables.
        """
        ds = SplitDataSetA(self.config, target_features=self.ds_extract.target_features)

        ds.process_targets()

        self.assertIsInstance(ds.training_sets["temp"], pl.DataFrame)
        self.assertEqual(ds.training_sets["temp"].shape[0], 116)
        self.assertEqual(ds.training_sets["temp"].shape[1], 57)

        self.assertIsInstance(ds.test_sets["temp"], pl.DataFrame)
        self.assertEqual(ds.test_sets["temp"].shape[0], 12)
        self.assertEqual(ds.test_sets["temp"].shape[1], 56)

        self.assertIsInstance(ds.training_sets["psal"], pl.DataFrame)
        self.assertEqual(ds.training_sets["psal"].shape[0], 126)
        self.assertEqual(ds.training_sets["psal"].shape[1], 57)

        self.assertIsInstance(ds.test_sets["psal"], pl.DataFrame)
        self.assertEqual(ds.test_sets["psal"].shape[0], 14)
        self.assertEqual(ds.test_sets["psal"].shape[1], 56)

        self.assertIsInstance(ds.training_sets["pres"], pl.DataFrame)
        self.assertEqual(ds.training_sets["pres"].shape[0], 110)
        self.assertEqual(ds.training_sets["pres"].shape[1], 57)

        self.assertIsInstance(ds.test_sets["pres"], pl.DataFrame)
        self.assertEqual(ds.test_sets["pres"].shape[0], 12)
        self.assertEqual(ds.test_sets["pres"].shape[1], 56)

    def test_default_test_set_fraction(self):
        """
        Check that the default test_set_fraction (0.1) is used when
        the configuration does not provide a specific value.
        """
        ds = SplitDataSetA(self.config, target_features=self.ds_extract.target_features)
        ds.config.data["step_param_set"]["steps"]["split"]["test_set_fraction"] = None

        test_set_fraction = ds.get_test_set_fraction()
        self.assertEqual(test_set_fraction, 0.1)

    def test_default_k_fold(self):
        """
        Check that the default k_fold value (10) is used when the
        configuration does not provide a specific value.
        """
        ds = SplitDataSetA(self.config, target_features=self.ds_extract.target_features)
        ds.config.data["step_param_set"]["steps"]["split"]["k_fold"] = None

        k_fold = ds.get_k_fold()
        self.assertEqual(k_fold, 10)

    def test_write_training_sets(self):
        """
        Confirm that training sets for each target variable are
        successfully written to their respective parquet files,
        and clean up the created files afterwards.
        """
        ds = SplitDataSetA(self.config, target_features=self.ds_extract.target_features)
        ds.process_targets()

        data_path = Path(__file__).resolve().parent / "data" / "training"
        # Ensure the directory exists
        data_path.mkdir(parents=True, exist_ok=True)

        ds.output_file_names["train"]["temp"] = str(
            data_path / "temp_train_set_temp.parquet"
        )
        ds.output_file_names["train"]["psal"] = str(
            data_path / "temp_train_set_psal.parquet"
        )
        ds.output_file_names["train"]["pres"] = str(
            data_path / "temp_train_set_pres.parquet"
        )

        ds.write_training_sets()

        self.assertTrue(os.path.exists(ds.output_file_names["train"]["temp"]))
        self.assertTrue(os.path.exists(ds.output_file_names["train"]["psal"]))
        self.assertTrue(os.path.exists(ds.output_file_names["train"]["pres"]))

        os.remove(ds.output_file_names["train"]["temp"])
        os.remove(ds.output_file_names["train"]["psal"])
        os.remove(ds.output_file_names["train"]["pres"])

    def test_write_empty_training_sets(self):
        """
        Ensure that writing empty training sets raises a ValueError.
        """
        ds = SplitDataSetA(self.config, target_features=self.ds_extract.target_features)
        ds.process_targets()
        # Set training_sets to None to simulate an empty state or error condition
        ds.training_sets = None
        with self.assertRaises(ValueError):
            ds.write_training_sets()

    def test_write_test_sets(self):
        """
        Confirm that test sets for each target variable are
        successfully written to their respective parquet files,
        and clean up the created files afterwards.
        """
        ds = SplitDataSetA(self.config, target_features=self.ds_extract.target_features)
        ds.process_targets()

        data_path = Path(__file__).resolve().parent / "data" / "training"
        # Ensure the directory exists
        data_path.mkdir(parents=True, exist_ok=True)

        ds.output_file_names["test"]["temp"] = str(
            data_path / "temp_test_set_temp.parquet"
        )
        ds.output_file_names["test"]["psal"] = str(
            data_path / "temp_test_set_psal.parquet"
        )
        ds.output_file_names["test"]["pres"] = str(
            data_path / "temp_test_set_pres.parquet"
        )

        ds.write_test_sets()

        self.assertTrue(os.path.exists(ds.output_file_names["test"]["temp"]))
        self.assertTrue(os.path.exists(ds.output_file_names["test"]["psal"]))
        self.assertTrue(os.path.exists(ds.output_file_names["test"]["pres"]))

        os.remove(ds.output_file_names["test"]["temp"])
        os.remove(ds.output_file_names["test"]["psal"])
        os.remove(ds.output_file_names["test"]["pres"])

    def test_write_empty_test_sets(self):
        """
        Ensure that writing empty test sets raises a ValueError.
        """
        ds = SplitDataSetA(self.config, target_features=self.ds_extract.target_features)
        ds.process_targets()
        # Set test_sets to None to simulate an empty state or error condition
        ds.test_sets = None
        with self.assertRaises(ValueError):
            ds.write_test_sets()

    def test_write_data_sets(self):
        """
        Verify that calling write_data_sets successfully writes
        both training and test sets for all target variables to
        parquet files, and clean up the created files afterwards.
        """
        ds = SplitDataSetA(self.config, target_features=self.ds_extract.target_features)
        ds.process_targets()

        data_path = Path(__file__).resolve().parent / "data" / "training"
        # Ensure the directory exists
        data_path.mkdir(parents=True, exist_ok=True)

        ds.output_file_names["train"]["temp"] = str(
            data_path / "temp_train_set_temp.parquet"
        )
        ds.output_file_names["train"]["psal"] = str(
            data_path / "temp_train_set_psal.parquet"
        )
        ds.output_file_names["train"]["pres"] = str(
            data_path / "temp_train_set_pres.parquet"
        )
        ds.output_file_names["test"]["temp"] = str(
            data_path / "temp_test_set_temp.parquet"
        )
        ds.output_file_names["test"]["psal"] = str(
            data_path / "temp_test_set_psal.parquet"
        )
        ds.output_file_names["test"]["pres"] = str(
            data_path / "temp_test_set_pres.parquet"
        )

        ds.write_data_sets()

        self.assertTrue(os.path.exists(ds.output_file_names["train"]["temp"]))
        self.assertTrue(os.path.exists(ds.output_file_names["train"]["psal"]))
        self.assertTrue(os.path.exists(ds.output_file_names["train"]["pres"]))
        self.assertTrue(os.path.exists(ds.output_file_names["test"]["temp"]))
        self.assertTrue(os.path.exists(ds.output_file_names["test"]["psal"]))
        self.assertTrue(os.path.exists(ds.output_file_names["test"]["pres"]))

        os.remove(ds.output_file_names["train"]["temp"])
        os.remove(ds.output_file_names["train"]["psal"])
        os.remove(ds.output_file_names["train"]["pres"])
        os.remove(ds.output_file_names["test"]["temp"])
        os.remove(ds.output_file_names["test"]["psal"])
        os.remove(ds.output_file_names["test"]["pres"])


class TestSplitDataSetANegX5(unittest.TestCase):
    """
    A suite of unit tests for SplitDataSetA specifically using the
    'test_dataset_003.yaml' configuration, which likely involves
    different parameters or larger data, verifying correct data
    splitting and handling.
    """

    def setUp(self):
        """
        Set up test environment and load data from previous steps
        (input, summary, select, locate, extract) to provide necessary
        dependencies for SplitDataSetA using a specific configuration.
        """
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

    def test_target_features_data(self):
        """
        Check that target features (extracted dataframes) are correctly
        loaded into the SplitDataSetA class upon initialization,
        verifying their type and dimensions for the specific config.
        """
        ds = SplitDataSetA(self.config, target_features=self.ds_extract.target_features)

        self.assertIsInstance(ds.target_features["temp"], pl.DataFrame)
        self.assertEqual(ds.target_features["temp"].shape[0], 831)
        self.assertEqual(ds.target_features["temp"].shape[1], 58)

        self.assertIsInstance(ds.target_features["psal"], pl.DataFrame)
        self.assertEqual(ds.target_features["psal"].shape[0], 903)
        self.assertEqual(ds.target_features["psal"].shape[1], 58)

        self.assertIsInstance(ds.target_features["pres"], pl.DataFrame)
        self.assertEqual(ds.target_features["pres"].shape[0], 783)
        self.assertEqual(ds.target_features["pres"].shape[1], 58)

    def test_split_features_data(self):
        """
        Verify the splitting of features into training and test sets
        for the given configuration, checking the resulting dimensions
        of the dataframes and ensuring row counts sum correctly.
        """
        ds = SplitDataSetA(self.config, target_features=self.ds_extract.target_features)

        ds.process_targets()

        self.assertIsInstance(ds.training_sets["temp"], pl.DataFrame)
        self.assertIsInstance(ds.test_sets["temp"], pl.DataFrame)
        self.assertEqual(ds.training_sets["temp"].shape[1], 57)
        self.assertEqual(ds.test_sets["temp"].shape[1], 56)
        self.assertEqual(
            ds.training_sets["temp"].shape[0] + ds.test_sets["temp"].shape[0], 831
        )

        self.assertIsInstance(ds.training_sets["psal"], pl.DataFrame)
        self.assertIsInstance(ds.test_sets["psal"], pl.DataFrame)
        self.assertEqual(ds.training_sets["psal"].shape[1], 57)
        self.assertEqual(ds.test_sets["psal"].shape[1], 56)
        self.assertEqual(
            ds.training_sets["psal"].shape[0] + ds.test_sets["psal"].shape[0], 903
        )

        self.assertIsInstance(ds.training_sets["pres"], pl.DataFrame)
        self.assertIsInstance(ds.test_sets["pres"], pl.DataFrame)
        self.assertEqual(ds.training_sets["pres"].shape[1], 57)
        self.assertEqual(ds.test_sets["pres"].shape[1], 56)
        self.assertEqual(
            ds.training_sets["pres"].shape[0] + ds.test_sets["pres"].shape[0], 783
        )

    def test_write_data_sets(self):
        """
        Verify that calling write_data_sets successfully writes
        both training and test sets for all target variables to
        parquet files, and clean up the created files afterwards.
        """
        ds = SplitDataSetA(self.config, target_features=self.ds_extract.target_features)
        ds.process_targets()

        data_path = Path(__file__).resolve().parent / "data" / "training"
        # Ensure the directory exists
        data_path.mkdir(parents=True, exist_ok=True)

        ds.output_file_names["train"]["temp"] = str(
            data_path / "temp_train_set_temp.parquet"
        )
        ds.output_file_names["train"]["psal"] = str(
            data_path / "temp_train_set_psal.parquet"
        )
        ds.output_file_names["train"]["pres"] = str(
            data_path / "temp_train_set_pres.parquet"
        )
        ds.output_file_names["test"]["temp"] = str(
            data_path / "temp_test_set_temp.parquet"
        )
        ds.output_file_names["test"]["psal"] = str(
            data_path / "temp_test_set_psal.parquet"
        )
        ds.output_file_names["test"]["pres"] = str(
            data_path / "temp_test_set_pres.parquet"
        )

        ds.write_data_sets()

        self.assertTrue(os.path.exists(ds.output_file_names["train"]["temp"]))
        self.assertTrue(os.path.exists(ds.output_file_names["train"]["psal"]))
        self.assertTrue(os.path.exists(ds.output_file_names["train"]["pres"]))
        self.assertTrue(os.path.exists(ds.output_file_names["test"]["temp"]))
        self.assertTrue(os.path.exists(ds.output_file_names["test"]["psal"]))
        self.assertTrue(os.path.exists(ds.output_file_names["test"]["pres"]))

        os.remove(ds.output_file_names["train"]["temp"])
        os.remove(ds.output_file_names["train"]["psal"])
        os.remove(ds.output_file_names["train"]["pres"])
        os.remove(ds.output_file_names["test"]["temp"])
        os.remove(ds.output_file_names["test"]["psal"])
        os.remove(ds.output_file_names["test"]["pres"])
