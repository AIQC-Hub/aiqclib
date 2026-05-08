"""
This module contains unit tests for the ExtractDataSetAll class,
verifying its functionality in gathering and processing features from
various classification pipeline steps (input, summary, select, locate).
"""

import os
import pytest
from pathlib import Path

import polars as pl

from aiqclib.classify.step5_extract_features.dataset_all import ExtractDataSetAll
from aiqclib.common.config.classify_config import ClassificationConfig
from aiqclib.common.loader.classify_loader import (
    load_classify_step1_input_dataset,
    load_classify_step2_summary_dataset,
    load_classify_step3_select_dataset,
    load_classify_step4_locate_dataset,
)


class TestExtractDataSetAll:
    """
    A suite of tests verifying that the ExtractDataSetA class gathers
    and outputs extracted features from multiple prior steps (input, summary,
    select, locate).
    """

    def _setup_configs(self):
        self.configs = []
        for x in self.config_file_paths:
            c = ClassificationConfig(str(x))
            c.select("NRT_BO_001")
            self.configs.append(c)

    def _setup_input_datasets(self):
        self.ds_input = []
        for x in self.configs:
            ds = load_classify_step1_input_dataset(x)
            ds.input_file_name = str(self.test_data_file)
            ds.read_input_data()
            self.ds_input.append(ds)

    def _setup_summary_datasets(self):
        self.ds_summary = []
        for x, y in zip(self.configs, self.ds_input):
            ds = load_classify_step2_summary_dataset(x, input_data=y.input_data)
            ds.calculate_stats()
            self.ds_summary.append(ds)

    def _setup_select_datasets(self):
        self.ds_select = []
        for x, y in zip(self.configs, self.ds_input):
            ds = load_classify_step3_select_dataset(x, input_data=y.input_data)
            ds.label_profiles()
            self.ds_select.append(ds)

    def _setup_locate_datasets(self):
        self.ds_locate = []
        for x, y, z in zip(self.configs, self.ds_input, self.ds_select):
            ds = load_classify_step4_locate_dataset(
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
                / "test_classify_001.yaml"
            ),
            (
                Path(__file__).resolve().parent
                / "data"
                / "config"
                / "test_classify_002.yaml"
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
        Test that the output file names dictionary is correctly populated
        based on the configuration.
        """
        ds = ExtractDataSetAll(self.configs[idx])
        assert (
            "/path/to/data_1/nrt_bo_001/extract/extracted_features_classify_temp.parquet"
            == str(ds.output_file_names["temp"])
        )
        assert (
            "/path/to/data_1/nrt_bo_001/extract/extracted_features_classify_psal.parquet"
            == str(ds.output_file_names["psal"])
        )
        assert (
            "/path/to/data_1/nrt_bo_001/extract/extracted_features_classify_pres.parquet"
            == str(ds.output_file_names["pres"])
        )

    @pytest.mark.parametrize("idx", range(2))
    def test_step_name(self, idx):
        """
        Verify that the 'step_name' attribute of the ExtractDataSetAll instance
        is correctly set to 'extract'.
        """
        ds = ExtractDataSetAll(self.configs[idx])
        assert ds.step_name == "extract"

    @pytest.mark.parametrize("idx", range(2))
    def test_init_arguments(self, idx):
        """
        Test that the ExtractDataSetAll class correctly initializes with
        provided input data, selected profiles, selected rows, and summary statistics,
        and that these dataframes have the expected shapes.
        """
        ds = ExtractDataSetAll(
            self.configs[idx],
            input_data=self.ds_input[idx].input_data,
            selected_profiles=self.ds_select[idx].selected_profiles,
            selected_rows=self.ds_locate[idx].selected_rows,
            summary_stats=self.ds_summary[idx].summary_stats,
        )

        assert isinstance(ds.input_data, pl.DataFrame)
        assert ds.input_data.shape[0] == 19480
        assert ds.input_data.shape[1] == 30

        assert isinstance(ds.summary_stats, pl.DataFrame)
        assert ds.summary_stats.shape[0] == 425
        assert ds.summary_stats.shape[1] == 12

        assert isinstance(ds.selected_profiles, pl.DataFrame)
        assert ds.selected_profiles.shape[0] == 84
        assert ds.selected_profiles.shape[1] == 8

        assert isinstance(ds.selected_rows["temp"], pl.DataFrame)
        assert ds.selected_rows["temp"].shape[0] == 19480
        assert ds.selected_rows["temp"].shape[1] == 9

        assert isinstance(ds.selected_rows["psal"], pl.DataFrame)
        assert ds.selected_rows["psal"].shape[0] == 19480
        assert ds.selected_rows["psal"].shape[1] == 9

        assert isinstance(ds.selected_rows["pres"], pl.DataFrame)
        assert ds.selected_rows["pres"].shape[0] == 19480
        assert ds.selected_rows["pres"].shape[1] == 9

    @pytest.mark.parametrize("idx", range(2))
    def test_location_features(self, idx):
        """
        Test that the `process_targets` method correctly generates and
        stores extracted features for all configured targets (temp, psal, pres)
        with the expected DataFrame shapes.
        """
        ds = ExtractDataSetAll(
            self.configs[idx],
            input_data=self.ds_input[idx].input_data,
            selected_profiles=self.ds_select[idx].selected_profiles,
            selected_rows=self.ds_locate[idx].selected_rows,
            summary_stats=self.ds_summary[idx].summary_stats,
        )

        ds.process_targets()

        assert isinstance(ds.target_features["temp"], pl.DataFrame)
        assert ds.target_features["temp"].shape[0] == 19480
        assert ds.target_features["temp"].shape[1] == 56

        assert isinstance(ds.target_features["psal"], pl.DataFrame)
        assert ds.target_features["psal"].shape[0] == 19480
        assert ds.target_features["psal"].shape[1] == 56

        assert isinstance(ds.target_features["pres"], pl.DataFrame)
        assert ds.target_features["pres"].shape[0] == 19480
        assert ds.target_features["pres"].shape[1] == 56

    @pytest.mark.parametrize("idx", range(2))
    def test_write_target_features(self, idx):
        """
        Test that the `write_target_features` method successfully writes
        the extracted features for configured targets to parquet files
        and that these files are created in the specified location.
        """
        ds = ExtractDataSetAll(
            self.configs[idx],
            input_data=self.ds_input[idx].input_data,
            selected_profiles=self.ds_select[idx].selected_profiles,
            selected_rows=self.ds_locate[idx].selected_rows,
            summary_stats=self.ds_summary[idx].summary_stats,
        )
        data_path = Path(__file__).resolve().parent / "data" / "extract"

        ds.output_file_names["temp"] = str(
            data_path / "temp_extracted_features_classify_temp.parquet"
        )
        ds.output_file_names["psal"] = str(
            data_path / "temp_extracted_features_classify_psal.parquet"
        )
        ds.output_file_names["pres"] = str(
            data_path / "temp_extracted_features_classify_pres.parquet"
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
        Test that calling `write_target_features` when `target_features`
        is empty (i.e., `process_targets` has not been called) raises a ValueError,
        ensuring proper validation.
        """
        ds = ExtractDataSetAll(
            self.configs[idx],
            input_data=self.ds_input[idx].input_data,
            selected_profiles=self.ds_select[idx].selected_profiles,
            selected_rows=self.ds_locate[idx].selected_rows,
            summary_stats=self.ds_summary[idx].summary_stats,
        )

        with pytest.raises(ValueError):
            ds.write_target_features()
