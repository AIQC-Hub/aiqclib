"""
This module contains unit tests for the ExtractDataSetA class,
verifying its functionality in processing and extracting features from
various intermediate datasets (input, summary, select, and locate)
generated in previous data quality control steps.
"""

import os
import unittest
import pytest
from pathlib import Path

import polars as pl

from aiqclib.common.config.dataset_config import DataSetConfig
from aiqclib.common.loader.dataset_loader import (
    load_step1_input_dataset,
    load_step2_summary_dataset,
    load_step3_select_dataset,
    load_step4_locate_dataset,
)
from aiqclib.prepare.step5_extract_features.dataset_a import ExtractDataSetA


class TestExtractDataSetA:
    """
    A suite of tests verifying that the ExtractDataSetA class gathers
    and outputs extracted features from multiple prior steps (input, summary,
    select, locate).
    """

    def _setup_configs(self):
        self.configs = []
        for x in self.config_file_paths:
            c = DataSetConfig(str(x))
            c.select("NRT_BO_001")
            self.configs.append(c)

    def _setup_input_datasets(self):
        self.ds_input = []
        for x in self.configs:
            ds = load_step1_input_dataset(x)
            ds.input_file_name = str(self.test_data_file)
            ds.read_input_data()
            self.ds_input.append(ds)

    def _setup_summary_datasets(self):
        self.ds_summary = []
        for x, y in zip(self.configs, self.ds_input):
            ds = load_step2_summary_dataset(x, input_data=y.input_data)
            ds.calculate_stats()
            self.ds_summary.append(ds)

    def _setup_select_datasets(self):
        self.ds_select = []
        for x, y in zip(self.configs, self.ds_input):
            ds = load_step3_select_dataset(x, input_data=y.input_data)
            ds.label_profiles()
            self.ds_select.append(ds)

    def _setup_locate_datasets(self):
        self.ds_locate = []
        for x, y, z in zip(self.configs, self.ds_input, self.ds_select):
            ds = load_step4_locate_dataset(
                x, input_data=y.input_data, selected_profiles=z.selected_profiles
            )
            ds.process_targets()
            self.ds_locate.append(ds)

    @pytest.fixture(autouse=True)
    def setup_datasets(self):
        """
        Set up test environment and load input, summary, select, and locate data.
        Initializes `DataSetConfig` and pre-processes data using prior steps
        to prepare for `ExtractDataSetA` testing.
        """
        self.config_file_paths = [
            (
                Path(__file__).resolve().parent
                / "data"
                / "config"
                / "test_dataset_001.yaml"
            ),
            (
                Path(__file__).resolve().parent
                / "data"
                / "config"
                / "test_dataset_004.yaml"
            ),
        ]
        self.test_data_file = (
            Path(__file__).resolve().parent
            / "data"
            / "input"
            / "nrt_cora_bo_test.parquet"
        )

        self._setup_configs()
        self._setup_input_datasets()
        self._setup_summary_datasets()
        self._setup_select_datasets()
        self._setup_locate_datasets()

    @pytest.mark.parametrize("idx", range(2))
    def test_output_file_names(self, idx):
        """
        Check that the output file names are set according to the configuration.
        Verifies the `output_file_names` attribute is correctly initialized.
        """
        ds = ExtractDataSetA(self.configs[idx])
        assert (
            "/path/to/data_1/nrt_bo_001/extract/extracted_features_temp.parquet"
            == str(ds.output_file_names["temp"])
        )
        assert (
            "/path/to/data_1/nrt_bo_001/extract/extracted_features_psal.parquet"
            == str(ds.output_file_names["psal"])
        )

    @pytest.mark.parametrize("idx", range(2))
    def test_step_name(self, idx):
        """
        Ensure the step name is set to 'extract'.
        Confirms the `step_name` attribute matches the expected value.
        """
        ds = ExtractDataSetA(self.configs[idx])
        assert ds.step_name == "extract"

    @pytest.mark.parametrize("idx", range(2))
    def test_init_arguments(self, idx):
        """
        Validate that input data, selected profiles, target rows, and summary stats are set.
        Checks if the `ExtractDataSetA` constructor correctly initializes and filters
        its internal data attributes based on provided inputs.
        """
        ds = ExtractDataSetA(
            self.configs[idx],
            input_data=self.ds_input[idx].input_data,
            selected_profiles=self.ds_select[idx].selected_profiles,
            selected_rows=self.ds_locate[idx].selected_rows,
            summary_stats=self.ds_summary[idx].summary_stats,
        )

        assert isinstance(ds.input_data, pl.DataFrame)
        assert ds.input_data.shape[0] == 132342
        assert ds.input_data.shape[1] == 30

        assert isinstance(ds.summary_stats, pl.DataFrame)
        assert ds.summary_stats.shape[0] == 2520
        assert ds.summary_stats.shape[1] == 12

        assert isinstance(ds.selected_profiles, pl.DataFrame)
        assert ds.selected_profiles.shape[0] == 50
        assert ds.selected_profiles.shape[1] == 8

        assert isinstance(ds.filtered_input, pl.DataFrame)
        assert ds.filtered_input.shape[0] == 10683
        assert ds.filtered_input.shape[1] == 30

        assert isinstance(ds.selected_rows["temp"], pl.DataFrame)
        assert ds.selected_rows["temp"].shape[0] == 128
        assert ds.selected_rows["temp"].shape[1] == 9

        assert isinstance(ds.selected_rows["psal"], pl.DataFrame)
        assert ds.selected_rows["psal"].shape[0] == 140
        assert ds.selected_rows["psal"].shape[1] == 9

        assert isinstance(ds.selected_rows["pres"], pl.DataFrame)
        assert ds.selected_rows["pres"].shape[0] == 122
        assert ds.selected_rows["pres"].shape[1] == 9

    @pytest.mark.parametrize("idx", range(2))
    def test_location_features(self, idx):
        """
        Check that features are correctly processed for temp and psal targets.
        Tests the `process_targets` method to ensure the correct generation
        and dimensions of extracted features for different target variables.
        """
        ds = ExtractDataSetA(
            self.configs[idx],
            input_data=self.ds_input[idx].input_data,
            selected_profiles=self.ds_select[idx].selected_profiles,
            selected_rows=self.ds_locate[idx].selected_rows,
            summary_stats=self.ds_summary[idx].summary_stats,
        )

        ds.process_targets()

        assert isinstance(ds.target_features["temp"], pl.DataFrame)
        assert ds.target_features["temp"].shape[0] == 128
        assert ds.target_features["temp"].shape[1] == 58

        assert isinstance(ds.target_features["psal"], pl.DataFrame)
        assert ds.target_features["psal"].shape[0] == 140
        assert ds.target_features["psal"].shape[1] == 58

        assert isinstance(ds.target_features["pres"], pl.DataFrame)
        assert ds.target_features["pres"].shape[0] == 122
        assert ds.target_features["pres"].shape[1] == 58

    @pytest.mark.parametrize("idx", range(2))
    def test_write_target_features(self, idx):
        """
        Confirm that target features are written to parquet files as expected.
        Verifies the `write_target_features` method successfully creates
        output files and cleans them up.
        """
        ds = ExtractDataSetA(
            self.configs[idx],
            input_data=self.ds_input[idx].input_data,
            selected_profiles=self.ds_select[idx].selected_profiles,
            selected_rows=self.ds_locate[idx].selected_rows,
            summary_stats=self.ds_summary[idx].summary_stats,
        )

        data_path = Path(__file__).resolve().parent / "data" / "extract"
        ds.output_file_names["temp"] = str(
            data_path / "temp_extracted_features_temp.parquet"
        )
        ds.output_file_names["psal"] = str(
            data_path / "temp_extracted_features_psal.parquet"
        )
        ds.output_file_names["pres"] = str(
            data_path / "temp_extracted_features_pres.parquet"
        )

        ds.process_targets()
        ds.write_target_features()

        assert os.path.exists(ds.output_file_names["temp"])
        assert os.path.exists(ds.output_file_names["psal"])
        assert os.path.exists(ds.output_file_names["pres"])
        os.remove(ds.output_file_names["temp"])
        os.remove(ds.output_file_names["psal"])
        os.remove(ds.output_file_names["pres"])

    @pytest.mark.parametrize("idx", range(2))
    def test_write_no_target_features(self, idx):
        """
        Check that calling write_target_features with empty features raises ValueError.
        Ensures `write_target_features` handles cases where no features have been processed yet.
        """
        ds = ExtractDataSetA(
            self.configs[idx],
            input_data=self.ds_input[idx].input_data,
            selected_profiles=self.ds_select[idx].selected_profiles,
            selected_rows=self.ds_locate[idx].selected_rows,
            summary_stats=self.ds_summary[idx].summary_stats,
        )

        # target_features attribute is not populated without calling process_targets()
        with pytest.raises(ValueError):
            ds.write_target_features()


class TestExtractDataSetANegX5(unittest.TestCase):
    """
    A suite of tests verifying that the ExtractDataSetA class gathers
    and outputs extracted features from multiple prior steps (input, summary,
    select, locate). This suite uses a different configuration file
    (`test_dataset_003.yaml`) to test different dataset characteristics.
    """

    def setUp(self):
        """
        Set up test environment and load input, summary, select, and locate data.
        Initializes `DataSetConfig` using `test_dataset_003.yaml` and pre-processes
        data to prepare for `ExtractDataSetA` testing under different configurations.
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

    def test_init_arguments(self):
        """
        Validate that input data, selected profiles, target rows, and summary stats are set.
        Checks if the `ExtractDataSetA` constructor correctly initializes and filters
        its internal data attributes with data derived from `test_dataset_003.yaml` configuration.
        """
        ds = ExtractDataSetA(
            self.config,
            input_data=self.ds_input.input_data,
            selected_profiles=self.ds_select.selected_profiles,
            selected_rows=self.ds_locate.selected_rows,
            summary_stats=self.ds_summary.summary_stats,
        )

        self.assertIsInstance(ds.input_data, pl.DataFrame)
        self.assertEqual(ds.input_data.shape[0], 115872)
        self.assertEqual(ds.input_data.shape[1], 30)

        self.assertIsInstance(ds.summary_stats, pl.DataFrame)
        self.assertEqual(ds.summary_stats.shape[0], 2185)
        self.assertEqual(ds.summary_stats.shape[1], 12)

        self.assertIsInstance(ds.selected_profiles, pl.DataFrame)
        self.assertEqual(ds.selected_profiles.shape[0], 150)
        self.assertEqual(ds.selected_profiles.shape[1], 8)

        self.assertIsInstance(ds.filtered_input, pl.DataFrame)
        self.assertEqual(ds.filtered_input.shape[0], 26362)
        self.assertEqual(ds.filtered_input.shape[1], 30)

        self.assertIsInstance(ds.selected_rows["temp"], pl.DataFrame)
        self.assertEqual(ds.selected_rows["temp"].shape[0], 831)
        self.assertEqual(ds.selected_rows["temp"].shape[1], 9)

        self.assertIsInstance(ds.selected_rows["psal"], pl.DataFrame)
        self.assertEqual(ds.selected_rows["psal"].shape[0], 903)
        self.assertEqual(ds.selected_rows["psal"].shape[1], 9)

        self.assertIsInstance(ds.selected_rows["pres"], pl.DataFrame)
        self.assertEqual(ds.selected_rows["pres"].shape[0], 783)
        self.assertEqual(ds.selected_rows["pres"].shape[1], 9)

    def test_location_features(self):
        """
        Check that features are correctly processed for temp and psal targets.
        Tests the `process_targets` method to ensure the correct generation
        and dimensions of extracted features for different target variables
        using `test_dataset_003.yaml` configuration.
        """
        ds = ExtractDataSetA(
            self.config,
            input_data=self.ds_input.input_data,
            selected_profiles=self.ds_select.selected_profiles,
            selected_rows=self.ds_locate.selected_rows,
            summary_stats=self.ds_summary.summary_stats,
        )

        ds.process_targets()

        self.assertIsInstance(ds.target_features["temp"], pl.DataFrame)
        self.assertEqual(ds.target_features["temp"].shape[0], 831)
        self.assertEqual(ds.target_features["temp"].shape[1], 58)

        self.assertIsInstance(ds.target_features["psal"], pl.DataFrame)
        self.assertEqual(ds.target_features["psal"].shape[0], 903)
        self.assertEqual(ds.target_features["psal"].shape[1], 58)

        self.assertIsInstance(ds.target_features["pres"], pl.DataFrame)
        self.assertEqual(ds.target_features["pres"].shape[0], 783)
        self.assertEqual(ds.target_features["pres"].shape[1], 58)

    def test_write_target_features(self):
        """
        Confirm that target features are written to parquet files as expected.
        Verifies the `write_target_features` method successfully creates
        output files for the `test_dataset_003.yaml` configuration and cleans them up.
        """
        ds = ExtractDataSetA(
            self.config,
            input_data=self.ds_input.input_data,
            selected_profiles=self.ds_select.selected_profiles,
            selected_rows=self.ds_locate.selected_rows,
            summary_stats=self.ds_summary.summary_stats,
        )
        data_path = Path(__file__).resolve().parent / "data" / "extract"
        ds.output_file_names["temp"] = str(
            data_path / "temp_extracted_features_temp.parquet"
        )
        ds.output_file_names["psal"] = str(
            data_path / "temp_extracted_features_psal.parquet"
        )
        ds.output_file_names["pres"] = str(
            data_path / "temp_extracted_features_pres.parquet"
        )

        ds.process_targets()
        ds.write_target_features()

        self.assertTrue(os.path.exists(ds.output_file_names["temp"]))
        self.assertTrue(os.path.exists(ds.output_file_names["psal"]))
        self.assertTrue(os.path.exists(ds.output_file_names["pres"]))
        os.remove(ds.output_file_names["temp"])
        os.remove(ds.output_file_names["psal"])
        os.remove(ds.output_file_names["pres"])


class TestExtractDataSetAwithAll(unittest.TestCase):
    """
    A suite of tests verifying that the ExtractDataSetA class gathers
    and outputs extracted features from multiple prior steps (input, summary,
    select, locate). This suite uses a different configuration file
    (`test_dataset_005.yaml`) to test different dataset characteristics.
    """

    def setUp(self):
        """
        Set up test environment and load input, summary, select, and locate data.
        Initializes `DataSetConfig` using `test_dataset_003.yaml` and pre-processes
        data to prepare for `ExtractDataSetA` testing under different configurations.
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

    def test_init_arguments(self):
        """
        Validate that input data, selected profiles, target rows, and summary stats are set.
        Checks if the `ExtractDataSetA` constructor correctly initializes and filters
        its internal data attributes with data derived from `test_dataset_003.yaml` configuration.
        """
        ds = ExtractDataSetA(
            self.config,
            input_data=self.ds_input.input_data,
            selected_profiles=self.ds_select.selected_profiles,
            selected_rows=self.ds_locate.selected_rows,
            summary_stats=self.ds_summary.summary_stats,
        )

        self.assertIsInstance(ds.input_data, pl.DataFrame)
        self.assertEqual(ds.input_data.shape[0], 132342)
        self.assertEqual(ds.input_data.shape[1], 30)

        self.assertIsInstance(ds.summary_stats, pl.DataFrame)
        self.assertEqual(ds.summary_stats.shape[0], 2520)
        self.assertEqual(ds.summary_stats.shape[1], 12)

        self.assertIsInstance(ds.selected_profiles, pl.DataFrame)
        self.assertEqual(ds.selected_profiles.shape[0], 503)
        self.assertEqual(ds.selected_profiles.shape[1], 8)

        self.assertIsInstance(ds.filtered_input, pl.DataFrame)
        self.assertEqual(ds.filtered_input.shape[0], 132342)
        self.assertEqual(ds.filtered_input.shape[1], 30)

        self.assertIsInstance(ds.selected_rows["temp"], pl.DataFrame)
        self.assertEqual(ds.selected_rows["temp"].shape[0], 132342)
        self.assertEqual(ds.selected_rows["temp"].shape[1], 9)

        self.assertIsInstance(ds.selected_rows["psal"], pl.DataFrame)
        self.assertEqual(ds.selected_rows["psal"].shape[0], 132342)
        self.assertEqual(ds.selected_rows["psal"].shape[1], 9)

        self.assertIsInstance(ds.selected_rows["pres"], pl.DataFrame)
        self.assertEqual(ds.selected_rows["pres"].shape[0], 132342)
        self.assertEqual(ds.selected_rows["pres"].shape[1], 9)

    def test_location_features(self):
        """
        Check that features are correctly processed for temp and psal targets.
        Tests the `process_targets` method to ensure the correct generation
        and dimensions of extracted features for different target variables
        using `test_dataset_003.yaml` configuration.
        """
        ds = ExtractDataSetA(
            self.config,
            input_data=self.ds_input.input_data,
            selected_profiles=self.ds_select.selected_profiles,
            selected_rows=self.ds_locate.selected_rows,
            summary_stats=self.ds_summary.summary_stats,
        )

        ds.process_targets()

        self.assertIsInstance(ds.target_features["temp"], pl.DataFrame)
        self.assertEqual(ds.target_features["temp"].shape[0], 132342)
        self.assertEqual(ds.target_features["temp"].shape[1], 58)

        self.assertIsInstance(ds.target_features["psal"], pl.DataFrame)
        self.assertEqual(ds.target_features["psal"].shape[0], 132342)
        self.assertEqual(ds.target_features["psal"].shape[1], 58)

        self.assertIsInstance(ds.target_features["pres"], pl.DataFrame)
        self.assertEqual(ds.target_features["pres"].shape[0], 132342)
        self.assertEqual(ds.target_features["pres"].shape[1], 58)

    def test_write_target_features(self):
        """
        Confirm that target features are written to parquet files as expected.
        Verifies the `write_target_features` method successfully creates
        output files for the `test_dataset_003.yaml` configuration and cleans them up.
        """
        ds = ExtractDataSetA(
            self.config,
            input_data=self.ds_input.input_data,
            selected_profiles=self.ds_select.selected_profiles,
            selected_rows=self.ds_locate.selected_rows,
            summary_stats=self.ds_summary.summary_stats,
        )
        data_path = Path(__file__).resolve().parent / "data" / "extract"
        ds.output_file_names["temp"] = str(
            data_path / "temp_extracted_features_all_temp.parquet"
        )
        ds.output_file_names["psal"] = str(
            data_path / "temp_extracted_features_all_psal.parquet"
        )
        ds.output_file_names["pres"] = str(
            data_path / "temp_extracted_features_all_pres.parquet"
        )

        ds.process_targets()
        ds.write_target_features()

        self.assertTrue(os.path.exists(ds.output_file_names["temp"]))
        self.assertTrue(os.path.exists(ds.output_file_names["psal"]))
        self.assertTrue(os.path.exists(ds.output_file_names["pres"]))
        os.remove(ds.output_file_names["temp"])
        os.remove(ds.output_file_names["psal"])
        os.remove(ds.output_file_names["pres"])
