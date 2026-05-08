"""
This module contains unit tests for the LocateDataSetAll class,
verifying its functionality for selecting and processing data,
assigning output file names, and handling data persistence.
"""

import os
import unittest
from pathlib import Path

import polars as pl

from aiqclib.classify.step4_select_rows.dataset_all import LocateDataSetAll
from aiqclib.common.config.classify_config import ClassificationConfig
from aiqclib.common.loader.classify_loader import load_classify_step1_input_dataset
from aiqclib.common.loader.classify_loader import load_classify_step3_select_dataset


class TestLocateDataSetAll(unittest.TestCase):
    """
    A suite of tests for verifying the LocateDataSetAll class functionality,
    including row selection, data assignment, and file output.
    """

    def setUp(self):
        """
        Set up a test environment by loading a configuration file, reading
        input data, and selecting relevant profiles for subsequent tests.
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

        self.ds_input = load_classify_step1_input_dataset(self.config)
        self.ds_input.input_file_name = str(self.test_data_file)
        self.ds_input.read_input_data()

        self.ds_select = load_classify_step3_select_dataset(
            self.config, input_data=self.ds_input.input_data
        )
        self.ds_select.label_profiles()

    def test_output_file_names(self):
        """
        Validate that the output file names for each variable
        (e.g., temp, psal) are set as per the configuration settings.
        """
        ds = LocateDataSetAll(self.config)
        self.assertEqual(
            "/path/to/locate_1/nrt_bo_001/locate_folder_1/selected_rows_classify_temp.parquet",
            str(ds.output_file_names["temp"]),
        )
        self.assertEqual(
            "/path/to/locate_1/nrt_bo_001/locate_folder_1/selected_rows_classify_psal.parquet",
            str(ds.output_file_names["psal"]),
        )
        self.assertEqual(
            "/path/to/locate_1/nrt_bo_001/locate_folder_1/selected_rows_classify_pres.parquet",
            str(ds.output_file_names["pres"]),
        )

    def test_step_name(self):
        """
        Ensure that the step name within LocateDataSetAll is 'locate'.
        """
        ds = LocateDataSetAll(self.config)
        self.assertEqual(ds.step_name, "locate")

    def test_input_data_and_selected_profiles(self):
        """
        Confirm that input_data and selected_profiles are correctly
        assigned as Polars DataFrames.
        """
        ds = LocateDataSetAll(
            self.config,
            input_data=self.ds_input.input_data,
            selected_profiles=self.ds_select.selected_profiles,
        )

        self.assertIsInstance(ds.input_data, pl.DataFrame)
        self.assertEqual(ds.input_data.shape[0], 19480)
        self.assertEqual(ds.input_data.shape[1], 30)

        self.assertIsInstance(ds.selected_profiles, pl.DataFrame)
        self.assertEqual(ds.selected_profiles.shape[0], 84)
        self.assertEqual(ds.selected_profiles.shape[1], 8)

    def test_selected_rows(self):
        """
        Confirm that target rows are correctly compiled after
        merging positive and negative rows.
        """
        ds = LocateDataSetAll(
            self.config,
            input_data=self.ds_input.input_data,
            selected_profiles=self.ds_select.selected_profiles,
        )

        ds.process_targets()

        self.assertIsInstance(ds.selected_rows["temp"], pl.DataFrame)
        self.assertEqual(ds.selected_rows["temp"].shape[0], 19480)
        self.assertEqual(ds.selected_rows["temp"].shape[1], 9)

        self.assertIsInstance(ds.selected_rows["psal"], pl.DataFrame)
        self.assertEqual(ds.selected_rows["psal"].shape[0], 19480)
        self.assertEqual(ds.selected_rows["psal"].shape[1], 9)

        self.assertIsInstance(ds.selected_rows["pres"], pl.DataFrame)
        self.assertEqual(ds.selected_rows["pres"].shape[0], 19480)
        self.assertEqual(ds.selected_rows["pres"].shape[1], 9)

    def test_selected_rows_with_empty_input(self):
        """
        Check that a ValueError is raised if input data are absent
        when processing targets.
        """
        ds = LocateDataSetAll(
            self.config,
            input_data=None,
            selected_profiles=self.ds_select.selected_profiles,
        )
        with self.assertRaises(ValueError):
            ds.process_targets()

    def test_write_selected_rows(self):
        """
        Ensure that target rows are saved to Parquet files and
        verify the existence of these files.
        """
        ds = LocateDataSetAll(
            self.config,
            input_data=self.ds_input.input_data,
            selected_profiles=self.ds_select.selected_profiles,
        )
        data_path = Path(__file__).resolve().parent / "data" / "select"
        ds.output_file_names["temp"] = str(
            data_path / "temp_selected_rows_classify_temp.parquet"
        )
        ds.output_file_names["psal"] = str(
            data_path / "temp_selected_rows_classify_psal.parquet"
        )
        ds.output_file_names["pres"] = str(
            data_path / "temp_selected_rows_classify_pres.parquet"
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
        Check that a ValueError is raised if target rows are absent
        when attempting to write them.
        """
        ds = LocateDataSetAll(
            self.config,
            input_data=self.ds_input.input_data,
            selected_profiles=self.ds_select.selected_profiles,
        )

        with self.assertRaises(ValueError):
            ds.write_selected_rows()
