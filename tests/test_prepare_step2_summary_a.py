"""Unit tests for the SummaryDataSetA class, covering output file handling,
data loading, and statistical calculations."""

import os
import pytest
from pathlib import Path

import polars as pl

from aiqclib.common.config.dataset_config import DataSetConfig
from aiqclib.common.loader.dataset_loader import load_step1_input_dataset
from aiqclib.prepare.step2_calc_stats.dataset_a import SummaryDataSetA


class TestSelectDataSetA:
    """
    A suite of tests for verifying summary dataset operations in SummaryDataSetA.
    Ensures output filenames, data loading, and profile/statistical calculations
    function as expected.
    """

    def _setup_configs(self):
        self.configs = []
        for x in self.config_file_paths:
            c = DataSetConfig(str(x))
            c.select("NRT_BO_001")
            self.configs.append(c)

    def _setup_input_datasets(self):
        self.input_ds = []
        for x in self.configs:
            ds = load_step1_input_dataset(x)
            ds.input_file_name = str(self.test_data_file)
            ds.read_input_data()
            self.input_ds.append(ds)

    @pytest.fixture(autouse=True)
    def setup_input(self):
        """Set up test environment by loading configuration and input dataset."""
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

    @pytest.mark.parametrize("idx", range(2))
    def test_output_file_name(self, idx):
        """Verify that the output file name is set correctly based on the configuration."""
        ds = SummaryDataSetA(self.configs[idx])
        assert "/path/to/data_1/nrt_bo_001/summary/summary_stats.tsv" == str(
            ds.output_file_name
        )

    def test_default_output_file_name(self):
        """Verify that a default output file name is correctly set when `output_file_name` is not specified in the configuration."""
        config_file_path = str(
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_dataset_002.yaml"
        )
        config = DataSetConfig(config_file_path)
        config.select("NRT_BO_001")

        ds = SummaryDataSetA(config)
        assert (
            "/path/to/data_1/summary_dataset_folder/summary/summary_in_params.txt"
            == str(ds.output_file_name)
        )

    @pytest.mark.parametrize("idx", range(2))
    def test_step_name(self, idx):
        """Check that the step name attribute is accurately set to 'summary'."""
        ds = SummaryDataSetA(self.configs[idx])
        assert ds.step_name == "summary"

    @pytest.mark.parametrize("idx", range(2))
    def test_input_data(self, idx):
        """Confirm that `input_data` is correctly stored as a Polars DataFrame with expected dimensions."""
        ds = SummaryDataSetA(
            self.configs[idx], input_data=self.input_ds[idx].input_data
        )
        assert isinstance(ds.input_data, pl.DataFrame)
        assert ds.input_data.shape[0] == 132342
        assert ds.input_data.shape[1] == 30

    @pytest.mark.parametrize("idx", range(2))
    def test_global_stats(self, idx):
        """Check that `calculate_global_stats` returns a Polars DataFrame with the correct columns and row count."""
        ds = SummaryDataSetA(
            self.configs[idx], input_data=self.input_ds[idx].input_data
        )
        df = ds.calculate_global_stats("temp")
        assert isinstance(df, pl.DataFrame)
        assert df.shape[0] == 1
        assert df.shape[1] == 12

    @pytest.mark.parametrize("idx", range(2))
    def test_profile_stats(self, idx):
        """Check that `calculate_profile_stats` correctly processes grouped profiles and returns a DataFrame of expected dimensions."""
        ds = SummaryDataSetA(
            self.configs[idx], input_data=self.input_ds[idx].input_data
        )
        grouped_df = ds.input_data.group_by(ds.profile_col_names)
        df = ds.calculate_profile_stats(grouped_df, "temp")
        assert df.shape[0] == 503
        assert df.shape[1] == 12

    @pytest.mark.parametrize("idx", range(2))
    def test_summary_stats(self, idx):
        """Check that `calculate_stats` correctly populates `summary_stats` with the expected dimensions."""
        ds = SummaryDataSetA(
            self.configs[idx], input_data=self.input_ds[idx].input_data
        )
        ds.calculate_stats()
        assert ds.summary_stats.shape[0] == 2520
        assert ds.summary_stats.shape[1] == 12

    @pytest.mark.parametrize("idx", range(2))
    def test_write_summary_stats(self, idx):
        """Confirm that `summary_stats` are successfully written to a file and the file's existence is verified."""
        ds = SummaryDataSetA(
            self.configs[idx], input_data=self.input_ds[idx].input_data
        )
        ds.output_file_name = str(
            Path(__file__).resolve().parent
            / "data"
            / "summary"
            / "temp_summary_stats.tsv"
        )

        ds.calculate_stats()
        ds.write_summary_stats()
        assert os.path.exists(ds.output_file_name)
        os.remove(ds.output_file_name)

    @pytest.mark.parametrize("idx", range(2))
    def test_write_no_summary_stats(self, idx):
        """Ensure `ValueError` is raised if `write_summary_stats` is called when `summary_stats` is empty."""
        ds = SummaryDataSetA(
            self.configs[idx], input_data=self.input_ds[idx].input_data
        )

        with pytest.raises(ValueError):
            ds.write_summary_stats()

    @pytest.mark.parametrize("idx", range(2))
    def test_summary_stats_observation(self, idx):
        """Check that `create_summary_stats_observation` calculates observation-level summary statistics correctly with expected dimensions."""
        ds = SummaryDataSetA(
            self.configs[idx], input_data=self.input_ds[idx].input_data
        )
        ds.calculate_stats()
        ds.create_summary_stats_observation()
        assert ds.summary_stats_observation.shape[0] == 5
        assert ds.summary_stats_observation.shape[1] == 5

    @pytest.mark.parametrize("idx", range(2))
    def test_summary_stats_observation_without_stats_ds(self, idx):
        """Ensure `ValueError` is raised if `create_summary_stats_observation` is called when `summary_stats` is empty."""
        ds = SummaryDataSetA(
            self.configs[idx], input_data=self.input_ds[idx].input_data
        )
        with pytest.raises(ValueError):
            ds.create_summary_stats_observation()

    @pytest.mark.parametrize("idx", range(2))
    def test_summary_stats_profile(self, idx):
        """Check that `create_summary_stats_profile` calculates profile-level summary statistics correctly with expected dimensions."""
        ds = SummaryDataSetA(
            self.configs[idx], input_data=self.input_ds[idx].input_data
        )
        ds.calculate_stats()
        ds.create_summary_stats_profile()
        assert ds.summary_stats_profile.shape[0] == 27
        assert ds.summary_stats_profile.shape[1] == 6

    @pytest.mark.parametrize("idx", range(2))
    def test_summary_stats_profile_without_stats_ds(self, idx):
        """Ensure `ValueError` is raised if `create_summary_stats_profile` is called when `summary_stats` is empty."""
        ds = SummaryDataSetA(
            self.configs[idx], input_data=self.input_ds[idx].input_data
        )
        with pytest.raises(ValueError):
            ds.create_summary_stats_profile()
