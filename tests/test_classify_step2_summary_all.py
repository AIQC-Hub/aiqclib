"""Unit tests for the SummaryDataSetAll class.

This module contains tests for verifying the correct functionality of
SummaryDataSetAll, including output file name generation, data loading,
and the calculation of global and profile-specific statistics.
"""

import os
import pytest
from pathlib import Path

import polars as pl

from aiqclib.classify.step2_calc_stats.dataset_all import SummaryDataSetAll
from aiqclib.common.config.classify_config import ClassificationConfig
from aiqclib.common.loader.classify_loader import load_classify_step1_input_dataset


class TestSummaryDataSetAll:
    """
    A suite of tests for verifying summary dataset operations in SummaryDataSetAll.
    Ensures output filenames, data loading, and profile/statistical calculations
    function as expected.
    """

    def _setup_configs(self):
        self.configs = []
        for x in self.config_file_paths:
            c = ClassificationConfig(str(x))
            c.select("NRT_BO_001")
            self.configs.append(c)

    def _setup_input_datasets(self):
        self.input_ds = []
        for x in self.configs:
            ds = load_classify_step1_input_dataset(x)
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

    @pytest.mark.parametrize("idx", range(2))
    def test_output_file_name(self, idx):
        """Verify that the output file name is set correctly based on the configuration."""
        ds = SummaryDataSetAll(self.configs[idx])
        assert "/path/to/data_1/nrt_bo_001/summary/summary_stats_classify.tsv" == str(
            ds.output_file_name
        )

    @pytest.mark.parametrize("idx", range(2))
    def test_step_name(self, idx):
        """Check that the step name attribute is accurately set to 'summary'."""
        ds = SummaryDataSetAll(self.configs[idx])
        assert ds.step_name == "summary"

    @pytest.mark.parametrize("idx", range(2))
    def test_input_data(self, idx):
        """Confirm that input_data is correctly stored as a Polars DataFrame.

        Also verifies the dimensions (rows and columns) of the loaded data.
        """
        ds = SummaryDataSetAll(
            self.configs[idx], input_data=self.input_ds[idx].input_data
        )
        assert isinstance(ds.input_data, pl.DataFrame)
        assert ds.input_data.shape[0] == 19480
        assert ds.input_data.shape[1] == 30

    @pytest.mark.parametrize("idx", range(2))
    def test_global_stats(self, idx):
        """Check that calculate_global_stats returns correct columns and row count.

        Ensures the generated global statistics DataFrame has the expected structure.
        """
        ds = SummaryDataSetAll(
            self.configs[idx], input_data=self.input_ds[idx].input_data
        )
        df = ds.calculate_global_stats("temp")
        assert isinstance(df, pl.DataFrame)
        assert df.shape[0] == 1
        assert df.shape[1] == 12

    @pytest.mark.parametrize("idx", range(2))
    def test_profile_stats(self, idx):
        """Check that calculate_profile_stats processes grouped profiles correctly.

        Verifies the dimensions of the DataFrame containing profile-specific statistics.
        """
        ds = SummaryDataSetAll(
            self.configs[idx], input_data=self.input_ds[idx].input_data
        )
        grouped_df = ds.input_data.group_by(ds.profile_col_names)
        df = ds.calculate_profile_stats(grouped_df, "temp")
        assert df.shape[0] == 84
        assert df.shape[1] == 12

    @pytest.mark.parametrize("idx", range(2))
    def test_summary_stats(self, idx):
        """Check that calculate_stats populates summary_stats with correct dimensions.

        Ensures the final summary statistics DataFrame has the expected number
        of rows and columns after calculation.
        """
        ds = SummaryDataSetAll(
            self.configs[idx], input_data=self.input_ds[idx].input_data
        )
        ds.calculate_stats()
        assert ds.summary_stats.shape[0] == 425
        assert ds.summary_stats.shape[1] == 12

    @pytest.mark.parametrize("idx", range(2))
    def test_write_summary_stats(self, idx):
        """Confirm that summary statistics are written to file and file creation is verified.

        Creates a temporary file, writes the summary statistics, and then
        checks for its existence before cleaning up.
        """
        ds = SummaryDataSetAll(
            self.configs[idx], input_data=self.input_ds[idx].input_data
        )
        ds.output_file_name = str(
            Path(__file__).resolve().parent
            / "data"
            / "summary"
            / "temp_summary_stats_classify.tsv"
        )

        ds.calculate_stats()
        ds.write_summary_stats()
        assert os.path.exists(ds.output_file_name)
        os.remove(ds.output_file_name)

    @pytest.mark.parametrize("idx", range(2))
    def test_write_no_summary_stats(self, idx):
        """Ensure ValueError is raised if write_summary_stats is called with empty stats.

        Verifies that attempting to write statistics before they are calculated
        or if they are empty results in a ValueError.
        """
        ds = SummaryDataSetAll(
            self.configs[idx], input_data=self.input_ds[idx].input_data
        )

        with pytest.raises(ValueError):
            ds.write_summary_stats()
