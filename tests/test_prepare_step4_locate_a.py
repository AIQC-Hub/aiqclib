"""
Unit tests for the `LocateDataSetA` class, focusing on its functionality
for selecting and processing rows based on configured datasets, and handling
the output of processed data files.
"""

import os
import unittest
from pathlib import Path

import polars as pl

from aiqclib.common.config.dataset_config import DataSetConfig
from aiqclib.common.loader.dataset_loader import load_step1_input_dataset
from aiqclib.common.loader.dataset_loader import load_step3_select_dataset
from aiqclib.prepare.step4_select_rows.dataset_a import LocateDataSetA


class TestLocateDataSetA(unittest.TestCase):
    """
    A suite of tests for verifying the LocateDataSetA class functionality,
    including row selection, data assignment, and file output.
    """

    def setUp(self):
        """
        Set up the test environment for `TestLocateDataSetA`.
        Loads configuration, reads input data, and prepares selected profiles
        and target value dictionaries for testing.
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

        self.ds_select = load_step3_select_dataset(
            self.config, input_data=self.ds_input.input_data
        )
        self.ds_select.label_profiles()

        self.target_value_temp = {
            "flag": "temp_qc",
            "pos_flag_values": [
                4,
            ],
            "neg_flag_values": [
                1,
            ],
        }
        self.target_value_psal = {
            "flag": "psal_qc",
            "pos_flag_values": [
                4,
            ],
            "neg_flag_values": [
                1,
            ],
        }
        self.target_value_pres = {
            "flag": "pres_qc",
            "pos_flag_values": [
                4,
            ],
            "neg_flag_values": [
                1,
            ],
        }

    def test_output_file_names(self):
        """
        Tests that output file names are correctly generated based on
        the configuration settings for each variable.

        Note: The hardcoded absolute paths ("/path/to/locate_1/...")
        make this test brittle and dependent on a specific environment setup.
        It would be more robust to generate the expected paths dynamically
        based on the configuration's output directory, or mock the underlying
        path generation mechanism if `LocateDataSetA` relies on external factors.
        """
        ds = LocateDataSetA(self.config)
        self.assertEqual(
            "/path/to/locate_1/nrt_bo_001/locate_folder_1/selected_rows_temp.parquet",
            str(ds.output_file_names["temp"]),
        )
        self.assertEqual(
            "/path/to/locate_1/nrt_bo_001/locate_folder_1/selected_rows_psal.parquet",
            str(ds.output_file_names["psal"]),
        )

    def test_step_name(self):
        """
        Verifies that the `step_name` attribute of `LocateDataSetA`
        is correctly set to 'locate'.
        """
        ds = LocateDataSetA(self.config)
        self.assertEqual(ds.step_name, "locate")

    def test_input_data_and_selected_profiles(self):
        """
        Confirms that `input_data` and `selected_profiles` are correctly
        assigned as Polars DataFrames and have expected dimensions.
        """
        ds = LocateDataSetA(
            self.config,
            input_data=self.ds_input.input_data,
            selected_profiles=self.ds_select.selected_profiles,
        )

        self.assertIsInstance(ds.input_data, pl.DataFrame)
        self.assertEqual(ds.input_data.shape[0], 132342)
        self.assertEqual(ds.input_data.shape[1], 30)

        self.assertIsInstance(ds.selected_profiles, pl.DataFrame)
        self.assertEqual(ds.selected_profiles.shape[0], 50)
        self.assertEqual(ds.selected_profiles.shape[1], 8)

    def test_positive_rows(self):
        """
        Checks if positive rows are correctly identified and stored
        for 'temp', 'psal', and 'pres' based on their flags.
        """
        ds = LocateDataSetA(
            self.config,
            input_data=self.ds_input.input_data,
            selected_profiles=self.ds_select.selected_profiles,
        )
        ds.select_positive_rows("temp", self.target_value_temp)
        ds.select_positive_rows("psal", self.target_value_psal)
        ds.select_positive_rows("pres", self.target_value_pres)

        self.assertIsInstance(ds.positive_rows["temp"], pl.DataFrame)
        self.assertEqual(ds.positive_rows["temp"].shape[0], 64)
        self.assertEqual(ds.positive_rows["temp"].shape[1], 9)

        self.assertIsInstance(ds.positive_rows["psal"], pl.DataFrame)
        self.assertEqual(ds.positive_rows["psal"].shape[0], 70)
        self.assertEqual(ds.positive_rows["psal"].shape[1], 9)

        self.assertIsInstance(ds.positive_rows["pres"], pl.DataFrame)
        self.assertEqual(ds.positive_rows["pres"].shape[0], 61)
        self.assertEqual(ds.positive_rows["pres"].shape[1], 9)

    def test_negative_rows(self):
        """
        Checks if negative rows are correctly identified and stored
        after positive row selection for 'temp', 'psal', and 'pres'.
        """
        ds = LocateDataSetA(
            self.config,
            input_data=self.ds_input.input_data,
            selected_profiles=self.ds_select.selected_profiles,
        )
        ds.select_positive_rows("temp", self.target_value_temp)
        ds.select_negative_rows("temp", self.target_value_temp)

        ds.select_positive_rows("psal", self.target_value_psal)
        ds.select_negative_rows("psal", self.target_value_psal)

        ds.select_positive_rows("pres", self.target_value_pres)
        ds.select_negative_rows("pres", self.target_value_pres)

        self.assertIsInstance(ds.negative_rows["temp"], pl.DataFrame)
        self.assertEqual(ds.negative_rows["temp"].shape[0], 64)
        self.assertEqual(ds.negative_rows["temp"].shape[1], 8)

        self.assertIsInstance(ds.negative_rows["psal"], pl.DataFrame)
        self.assertEqual(ds.negative_rows["psal"].shape[0], 70)
        self.assertEqual(ds.negative_rows["psal"].shape[1], 8)

        self.assertIsInstance(ds.negative_rows["pres"], pl.DataFrame)
        self.assertEqual(ds.negative_rows["pres"].shape[0], 61)
        self.assertEqual(ds.negative_rows["pres"].shape[1], 8)

    def test_selected_rows(self):
        """
        Confirms that the combined 'selected_rows' for 'temp', 'psal',
        and 'pres' are correctly compiled and have expected dimensions.
        """
        ds = LocateDataSetA(
            self.config,
            input_data=self.ds_input.input_data,
            selected_profiles=self.ds_select.selected_profiles,
        )

        ds.process_targets()

        self.assertIsInstance(ds.selected_rows["temp"], pl.DataFrame)
        self.assertEqual(ds.selected_rows["temp"].shape[0], 128)
        self.assertEqual(ds.selected_rows["temp"].shape[1], 9)

        self.assertIsInstance(ds.selected_rows["psal"], pl.DataFrame)
        self.assertEqual(ds.selected_rows["psal"].shape[0], 140)
        self.assertEqual(ds.selected_rows["psal"].shape[1], 9)

        self.assertIsInstance(ds.selected_rows["pres"], pl.DataFrame)
        self.assertEqual(ds.selected_rows["pres"].shape[0], 122)
        self.assertEqual(ds.selected_rows["pres"].shape[1], 9)

    def test_write_selected_rows(self):
        """
        Verifies that the `write_selected_rows` method successfully creates
        Parquet files for 'temp', 'psal', and 'pres' and cleans them up.
        """
        ds = LocateDataSetA(
            self.config,
            input_data=self.ds_input.input_data,
            selected_profiles=self.ds_select.selected_profiles,
        )
        data_path = Path(__file__).resolve().parent / "data" / "select"
        ds.output_file_names["temp"] = str(
            data_path / "temp_selected_rows_temp.parquet"
        )
        ds.output_file_names["psal"] = str(
            data_path / "temp_selected_rows_psal.parquet"
        )
        ds.output_file_names["pres"] = str(
            data_path / "temp_selected_rows_pres.parquet"
        )

        ds.process_targets()
        ds.write_selected_rows()

        self.assertTrue(os.path.exists(ds.output_file_names["temp"]))
        self.assertTrue(os.path.exists(ds.output_file_names["psal"]))
        self.assertTrue(os.path.exists(ds.output_file_names["pres"]))
        os.remove(ds.output_file_names["temp"])
        os.remove(ds.output_file_names["psal"])
        os.remove(ds.output_file_names["pres"])

    def test_write_no_selected_rows(self):
        """
        Ensures that `write_selected_rows` raises a ValueError when
        `selected_rows` are not yet populated.
        """
        ds = LocateDataSetA(
            self.config,
            input_data=self.ds_input.input_data,
            selected_profiles=self.ds_select.selected_profiles,
        )

        with self.assertRaises(ValueError):
            ds.write_selected_rows()


class TestLocateDataSetANegX5(unittest.TestCase):
    """
    A suite of tests ensuring the LocateDataSetA class operates correctly
    for selecting and labeling profiles, as well as writing results to disk.
    This test suite specifically uses a configuration (`test_dataset_003.yaml`)
    that includes a `neg_x_multiplier` setting, which affects the number
    of negative rows selected.
    """

    def setUp(self):
        """
        Set up the test environment for `TestLocateDataSetANegX5`.
        Loads a specific configuration (`test_dataset_003.yaml`), reads
        input data, and prepares selected profiles and target value
        dictionaries for tests focused on negative row multiplication.
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

        self.ds_select = load_step3_select_dataset(
            self.config, input_data=self.ds_input.input_data
        )
        self.ds_select.label_profiles()

        self.target_value_temp = {
            "flag": "temp_qc",
            "pos_flag_values": [
                4,
            ],
            "neg_flag_values": [
                1,
            ],
        }
        self.target_value_psal = {
            "flag": "psal_qc",
            "pos_flag_values": [
                4,
            ],
            "neg_flag_values": [
                1,
            ],
        }
        self.target_value_pres = {
            "flag": "pres_qc",
            "pos_flag_values": [
                4,
            ],
            "neg_flag_values": [
                1,
            ],
        }

    def test_positive_rows(self):
        """
        Checks that positive rows are correctly identified and stored
        for 'temp', 'psal', and 'pres' based on their flags.
        (Expected counts are the same as TestLocateDataSetA as positive
        selection logic is not affected by `neg_x_multiplier`).
        """
        ds = LocateDataSetA(
            self.config,
            input_data=self.ds_input.input_data,
            selected_profiles=self.ds_select.selected_profiles,
        )
        ds.select_positive_rows("temp", self.target_value_temp)
        ds.select_positive_rows("psal", self.target_value_psal)
        ds.select_positive_rows("pres", self.target_value_pres)

        self.assertIsInstance(ds.positive_rows["temp"], pl.DataFrame)
        self.assertEqual(ds.positive_rows["temp"].shape[0], 64)
        self.assertEqual(ds.positive_rows["temp"].shape[1], 9)

        self.assertIsInstance(ds.positive_rows["psal"], pl.DataFrame)
        self.assertEqual(ds.positive_rows["psal"].shape[0], 70)
        self.assertEqual(ds.positive_rows["psal"].shape[1], 9)

        self.assertIsInstance(ds.positive_rows["pres"], pl.DataFrame)
        self.assertEqual(ds.positive_rows["pres"].shape[0], 61)
        self.assertEqual(ds.positive_rows["pres"].shape[1], 9)

    def test_negative_rows(self):
        """
        Checks that negative rows are correctly identified and stored,
        taking into account the `neg_x_multiplier` from the config,
        which should result in a larger number of negative rows compared
        to the default configuration.
        """
        ds = LocateDataSetA(
            self.config,
            input_data=self.ds_input.input_data,
            selected_profiles=self.ds_select.selected_profiles,
        )
        ds.select_positive_rows("temp", self.target_value_temp)
        ds.select_negative_rows("temp", self.target_value_temp)

        ds.select_positive_rows("psal", self.target_value_psal)
        ds.select_negative_rows("psal", self.target_value_psal)

        ds.select_positive_rows("pres", self.target_value_pres)
        ds.select_negative_rows("pres", self.target_value_pres)

        self.assertIsInstance(ds.negative_rows["temp"], pl.DataFrame)
        self.assertEqual(ds.negative_rows["temp"].shape[0], 767)
        self.assertEqual(ds.negative_rows["temp"].shape[1], 8)

        self.assertIsInstance(ds.negative_rows["psal"], pl.DataFrame)
        self.assertEqual(ds.negative_rows["psal"].shape[0], 833)
        self.assertEqual(ds.negative_rows["psal"].shape[1], 8)

        self.assertIsInstance(ds.negative_rows["pres"], pl.DataFrame)
        self.assertEqual(ds.negative_rows["pres"].shape[0], 722)
        self.assertEqual(ds.negative_rows["pres"].shape[1], 8)

    def test_selected_rows(self):
        """
        Confirms that the combined 'selected_rows' for 'temp', 'psal',
        and 'pres' are correctly compiled, including the multiplied
        negative rows, and have expected total dimensions.
        """
        ds = LocateDataSetA(
            self.config,
            input_data=self.ds_input.input_data,
            selected_profiles=self.ds_select.selected_profiles,
        )

        ds.process_targets()

        self.assertIsInstance(ds.selected_rows["temp"], pl.DataFrame)
        self.assertEqual(ds.selected_rows["temp"].shape[0], 831)
        self.assertEqual(ds.selected_rows["temp"].shape[1], 9)

        self.assertIsInstance(ds.selected_rows["psal"], pl.DataFrame)
        self.assertEqual(ds.selected_rows["psal"].shape[0], 903)
        self.assertEqual(ds.selected_rows["psal"].shape[1], 9)

        self.assertIsInstance(ds.selected_rows["pres"], pl.DataFrame)
        self.assertEqual(ds.selected_rows["pres"].shape[0], 783)
        self.assertEqual(ds.selected_rows["pres"].shape[1], 9)

    def test_write_selected_rows(self):
        """
        Verifies that the `write_selected_rows` method successfully creates
        Parquet files with the multiplied negative rows and cleans them up.
        """
        ds = LocateDataSetA(
            self.config,
            input_data=self.ds_input.input_data,
            selected_profiles=self.ds_select.selected_profiles,
        )
        data_path = Path(__file__).resolve().parent / "data" / "select"
        ds.output_file_names["temp"] = str(
            data_path / "temp_selected_rows_temp.parquet"
        )
        ds.output_file_names["psal"] = str(
            data_path / "temp_selected_rows_psal.parquet"
        )
        ds.output_file_names["pres"] = str(
            data_path / "temp_selected_rows_pres.parquet"
        )

        ds.process_targets()
        ds.write_selected_rows()

        self.assertTrue(os.path.exists(ds.output_file_names["temp"]))
        self.assertTrue(os.path.exists(ds.output_file_names["psal"]))
        self.assertTrue(os.path.exists(ds.output_file_names["pres"]))
        os.remove(ds.output_file_names["temp"])
        os.remove(ds.output_file_names["psal"])
        os.remove(ds.output_file_names["pres"])
