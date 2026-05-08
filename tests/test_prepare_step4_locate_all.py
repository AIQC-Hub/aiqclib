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
from aiqclib.prepare.step4_select_rows.dataset_all import LocateDataSetAll


class TestLocateDataSetAll(unittest.TestCase):
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

    def test_step_name(self):
        """
        Verifies that the `step_name` attribute of `LocateDataSetA`
        is correctly set to 'locate'.
        """
        ds = LocateDataSetAll(self.config)
        self.assertEqual(ds.step_name, "locate")

    def test_input_data_and_selected_profiles(self):
        """
        Confirms that `input_data` and `selected_profiles` are correctly
        assigned as Polars DataFrames and have expected dimensions.
        """
        ds = LocateDataSetAll(
            self.config,
            input_data=self.ds_input.input_data,
            selected_profiles=self.ds_select.selected_profiles,
        )

        self.assertIsInstance(ds.input_data, pl.DataFrame)
        self.assertEqual(ds.input_data.shape[0], 132342)
        self.assertEqual(ds.input_data.shape[1], 30)

        self.assertIsInstance(ds.selected_profiles, pl.DataFrame)
        self.assertEqual(ds.selected_profiles.shape[0], 503)
        self.assertEqual(ds.selected_profiles.shape[1], 8)

    def test_selected_rows(self):
        """
        Confirms that the combined 'selected_rows' for 'temp', 'psal',
        and 'pres' are correctly compiled and have expected dimensions.
        """
        ds = LocateDataSetAll(
            self.config,
            input_data=self.ds_input.input_data,
            selected_profiles=self.ds_select.selected_profiles,
        )

        ds.process_targets()

        self.assertIsInstance(ds.selected_rows["temp"], pl.DataFrame)
        self.assertEqual(ds.selected_rows["temp"].shape[0], 132342)
        self.assertEqual(ds.selected_rows["temp"].shape[1], 9)

        self.assertIsInstance(ds.selected_rows["psal"], pl.DataFrame)
        self.assertEqual(ds.selected_rows["psal"].shape[0], 132342)
        self.assertEqual(ds.selected_rows["psal"].shape[1], 9)

        self.assertIsInstance(ds.selected_rows["pres"], pl.DataFrame)
        self.assertEqual(ds.selected_rows["pres"].shape[0], 132342)
        self.assertEqual(ds.selected_rows["pres"].shape[1], 9)

    def test_write_selected_rows(self):
        """
        Verifies that the `write_selected_rows` method successfully creates
        Parquet files for 'temp', 'psal', and 'pres' and cleans them up.
        """
        ds = LocateDataSetAll(
            self.config,
            input_data=self.ds_input.input_data,
            selected_profiles=self.ds_select.selected_profiles,
        )
        data_path = Path(__file__).resolve().parent / "data" / "select"
        ds.output_file_names["temp"] = str(
            data_path / "temp_selected_rows_all_temp.parquet"
        )
        ds.output_file_names["psal"] = str(
            data_path / "temp_selected_rows_all_psal.parquet"
        )
        ds.output_file_names["pres"] = str(
            data_path / "temp_selected_rows_all_pres.parquet"
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
        ds = LocateDataSetAll(
            self.config,
            input_data=self.ds_input.input_data,
            selected_profiles=self.ds_select.selected_profiles,
        )

        with self.assertRaises(ValueError):
            ds.write_selected_rows()
