"""
Unit tests for verifying the correct loading and initialization of dataset classes
at various processing steps, using common loader functions.

These tests ensure that the dataset objects are correctly instantiated with the
expected step names and that any provided input data (e.g., from previous steps)
is properly assigned and retains its expected structure.
"""

import pytest
from pathlib import Path

import polars as pl

from aiqclib.common.config.dataset_config import DataSetConfig
from aiqclib.common.loader.dataset_loader import (
    load_step1_input_dataset,
    load_step2_summary_dataset,
    load_step3_select_dataset,
    load_step4_locate_dataset,
    load_step5_extract_dataset,
    load_step6_split_dataset,
)
from aiqclib.prepare.step1_read_input.dataset_a import InputDataSetA
from aiqclib.prepare.step2_calc_stats.dataset_a import SummaryDataSetA
from aiqclib.prepare.step3_select_profiles.dataset_a import SelectDataSetA
from aiqclib.prepare.step3_select_profiles.dataset_all import SelectDataSetAll
from aiqclib.prepare.step4_select_rows.dataset_a import LocateDataSetA
from aiqclib.prepare.step4_select_rows.dataset_all import LocateDataSetAll
from aiqclib.prepare.step5_extract_features.dataset_a import ExtractDataSetA
from aiqclib.prepare.step6_split_dataset.dataset_a import SplitDataSetA
from aiqclib.prepare.step6_split_dataset.dataset_all import SplitDataSetAll


class TestInputClassLoader:
    """
    Tests related to loading the InputDataSetA class.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """
        Define the path to the test config file and select a dataset
        prior to each test.
        """
        dir_config = Path(__file__).resolve().parent / "data" / "config"
        self.config_file_paths = [
            str(dir_config / "test_dataset_001.yaml"),
            str(dir_config / "test_dataset_005.yaml"),
        ]

    def _set_config(self, idx):
        self.config = DataSetConfig(str(self.config_file_paths[idx]))
        self.config.select("NRT_BO_001")

    @pytest.mark.parametrize("idx", range(2))
    def test_load_dataset_valid_config(self, idx):
        """
        Check that load_step1_input_dataset returns an InputDataSetA instance with
        the expected step name.
        """
        self._set_config(idx)
        ds = load_step1_input_dataset(self.config)
        assert isinstance(ds, InputDataSetA)
        assert ds.step_name == "input"

    def test_load_input_class_with_invalid_config(self):
        """
        Ensure that an invalid input class name raises a ValueError.
        """
        self._set_config(0)
        self.config.data["step_class_set"]["steps"]["input"] = "InvalidClass"
        with pytest.raises(ValueError):
            _ = load_step1_input_dataset(self.config)


class TestSummaryClassLoader:
    """
    Tests related to loading the SummaryDataSetA class.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """
        Define the path to the test config file and select a dataset
        prior to each test.
        """
        dir_config = Path(__file__).resolve().parent / "data" / "config"
        self.config_file_paths = [
            str(dir_config / "test_dataset_001.yaml"),
            str(dir_config / "test_dataset_005.yaml"),
        ]

        self.test_data_file = (
            Path(__file__).resolve().parent
            / "data"
            / "input"
            / "nrt_cora_bo_test.parquet"
        )

    def _set_config(self, idx):
        self.config = DataSetConfig(str(self.config_file_paths[idx]))
        self.config.select("NRT_BO_001")

    @pytest.mark.parametrize("idx", range(2))
    def test_load_dataset_valid_config(self, idx):
        """
        Check that load_step2_summary_dataset returns a SummaryDataSetA instance
        with the correct step name.
        """
        self._set_config(idx)
        ds = load_step2_summary_dataset(self.config)
        assert isinstance(ds, SummaryDataSetA)
        assert ds.step_name == "summary"

    @pytest.mark.parametrize("idx", range(2))
    def test_load_dataset_input_data(self, idx):
        """
        Check that load_step2_summary_dataset sets input_data properly
        when provided and retains its expected structure.
        """
        self._set_config(idx)
        ds_input = load_step1_input_dataset(self.config)
        ds_input.input_file_name = str(self.test_data_file)
        ds_input.read_input_data()

        ds = load_step2_summary_dataset(self.config, ds_input.input_data)
        assert isinstance(ds, SummaryDataSetA)
        assert isinstance(ds.input_data, pl.DataFrame)
        assert ds.input_data.shape[0] == 132342
        assert ds.input_data.shape[1] == 30


class TestSelectClassLoader:
    """
    Tests related to loading the SelectDataSetA class.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """
        Define the path to the test config file and select a dataset
        prior to each test.
        """
        dir_config = Path(__file__).resolve().parent / "data" / "config"
        self.config_file_paths = [
            str(dir_config / "test_dataset_001.yaml"),
            str(dir_config / "test_dataset_005.yaml"),
        ]

        self.select_classes = [
            SelectDataSetA,
            SelectDataSetAll,
        ]

        self.test_data_file = (
            Path(__file__).resolve().parent
            / "data"
            / "input"
            / "nrt_cora_bo_test.parquet"
        )

    def _set_config(self, idx):
        self.config = DataSetConfig(str(self.config_file_paths[idx]))
        self.config.select("NRT_BO_001")

    @pytest.mark.parametrize("idx", range(2))
    def test_load_dataset_valid_config(self, idx):
        """
        Check that load_step3_select_dataset returns a SelectDataSetA instance
        with the correct step name.
        """
        self._set_config(idx)
        ds = load_step3_select_dataset(self.config)
        assert isinstance(ds, self.select_classes[idx])
        assert ds.step_name == "select"

    @pytest.mark.parametrize("idx", range(2))
    def test_load_dataset_input_data(self, idx):
        """
        Check that load_step3_select_dataset sets input_data properly
        when provided and retains its expected structure.
        """
        self._set_config(idx)
        ds_input = load_step1_input_dataset(self.config)
        ds_input.input_file_name = str(self.test_data_file)
        ds_input.read_input_data()

        ds = load_step3_select_dataset(self.config, ds_input.input_data)
        assert isinstance(ds, self.select_classes[idx])
        assert isinstance(ds.input_data, pl.DataFrame)
        assert ds.input_data.shape[0] == 132342
        assert ds.input_data.shape[1] == 30


class TestLocateClassLoader:
    """
    Tests related to loading the LocateDataSetA class.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """
        Define the path to the test config file and select a dataset
        prior to each test.
        """
        dir_config = Path(__file__).resolve().parent / "data" / "config"
        self.config_file_paths = [
            str(dir_config / "test_dataset_001.yaml"),
            str(dir_config / "test_dataset_005.yaml"),
        ]

        self.locate_classes = [
            LocateDataSetA,
            LocateDataSetAll,
        ]

        self.selected_profiles = [50, 503]

        self.test_data_file = (
            Path(__file__).resolve().parent
            / "data"
            / "input"
            / "nrt_cora_bo_test.parquet"
        )

    def _set_config(self, idx):
        self.config = DataSetConfig(str(self.config_file_paths[idx]))
        self.config.select("NRT_BO_001")

    @pytest.mark.parametrize("idx", range(2))
    def test_load_dataset_valid_config(self, idx):
        """
        Check that load_step4_locate_dataset returns a LocateDataSetA instance
        with the correct step name.
        """
        self._set_config(idx)
        ds = load_step4_locate_dataset(self.config)
        assert isinstance(ds, self.locate_classes[idx])
        assert ds.step_name == "locate"

    @pytest.mark.parametrize("idx", range(2))
    def test_load_dataset_input_data_and_profiles(self, idx):
        """
        Check that load_step4_locate_dataset sets input_data and selected_profiles
        properly when provided and retains their expected structure.
        """
        self._set_config(idx)
        ds_input = load_step1_input_dataset(self.config)
        ds_input.input_file_name = str(self.test_data_file)
        ds_input.read_input_data()

        ds_select = load_step3_select_dataset(self.config, ds_input.input_data)
        ds_select.label_profiles()

        ds = load_step4_locate_dataset(
            self.config, ds_input.input_data, ds_select.selected_profiles
        )

        assert isinstance(ds, self.locate_classes[idx])

        assert isinstance(ds.input_data, pl.DataFrame)
        assert ds.input_data.shape[0] == 132342
        assert ds.input_data.shape[1] == 30

        assert isinstance(ds.selected_profiles, pl.DataFrame)
        assert ds.selected_profiles.shape[0] == self.selected_profiles[idx]
        assert ds.selected_profiles.shape[1] == 8


class TestExtractClassLoader:
    """
    Tests related to loading the ExtractDataSetA class.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """
        Define the path to the test config file and select a dataset
        prior to each test.
        """
        dir_config = Path(__file__).resolve().parent / "data" / "config"
        self.config_file_paths = [
            str(dir_config / "test_dataset_001.yaml"),
            str(dir_config / "test_dataset_005.yaml"),
        ]

        self.selected_profiles = [50, 503]
        self.filtered_inputs = [10683, 132342]
        self.selected_rows_temps = [128, 132342]
        self.selected_rows_psals = [140, 132342]

        self.test_data_file = (
            Path(__file__).resolve().parent
            / "data"
            / "input"
            / "nrt_cora_bo_test.parquet"
        )

    def _set_config(self, idx):
        self.config = DataSetConfig(str(self.config_file_paths[idx]))
        self.config.select("NRT_BO_001")

    @pytest.mark.parametrize("idx", range(2))
    def test_load_dataset_valid_config(self, idx):
        """
        Check that load_step5_extract_dataset returns an ExtractDataSetA instance
        with the correct step name.
        """
        self._set_config(idx)
        ds = load_step5_extract_dataset(self.config)
        assert isinstance(ds, ExtractDataSetA)
        assert ds.step_name == "extract"

    @pytest.mark.parametrize("idx", range(2))
    def test_load_dataset_input_data_and_profiles(self, idx):
        """
        Check that load_step5_extract_dataset correctly initializes the dataset
        with provided inputs and that derived attributes (e.g., `filtered_input`,
        `selected_rows`) are properly set and retain their expected structure.
        """
        self._set_config(idx)
        ds_input = load_step1_input_dataset(self.config)
        ds_input.input_file_name = str(self.test_data_file)
        ds_input.read_input_data()

        ds_select = load_step3_select_dataset(self.config, ds_input.input_data)
        ds_select.label_profiles()

        ds_summary = load_step2_summary_dataset(self.config, ds_input.input_data)
        ds_summary.calculate_stats()

        ds_locate = load_step4_locate_dataset(
            self.config, ds_input.input_data, ds_select.selected_profiles
        )
        ds_locate.process_targets()

        ds = load_step5_extract_dataset(
            self.config,
            ds_input.input_data,
            ds_select.selected_profiles,
            ds_locate.selected_rows,
            ds_summary.summary_stats,
        )

        assert isinstance(ds, ExtractDataSetA)

        assert isinstance(ds.input_data, pl.DataFrame)
        assert ds.input_data.shape[0] == 132342
        assert ds.input_data.shape[1] == 30

        assert isinstance(ds.summary_stats, pl.DataFrame)
        assert ds.summary_stats.shape[0] == 2520
        assert ds.summary_stats.shape[1] == 12

        assert isinstance(ds.selected_profiles, pl.DataFrame)
        assert ds.selected_profiles.shape[0] == self.selected_profiles[idx]
        assert ds.selected_profiles.shape[1] == 8

        assert isinstance(ds.filtered_input, pl.DataFrame)
        assert ds.filtered_input.shape[0] == self.filtered_inputs[idx]
        assert ds.filtered_input.shape[1] == 30

        assert isinstance(ds.selected_rows["temp"], pl.DataFrame)
        assert ds.selected_rows["temp"].shape[0] == self.selected_rows_temps[idx]
        assert ds.selected_rows["temp"].shape[1] == 9

        assert isinstance(ds.selected_rows["psal"], pl.DataFrame)
        assert ds.selected_rows["psal"].shape[0] == self.selected_rows_psals[idx]
        assert ds.selected_rows["psal"].shape[1] == 9


class TestSplitClassLoader:
    """
    Tests related to loading the SplitDataSetA class.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """
        Define the path to the test config file and select a dataset
        prior to each test.
        """
        dir_config = Path(__file__).resolve().parent / "data" / "config"
        self.config_file_paths = [
            str(dir_config / "test_dataset_001.yaml"),
            str(dir_config / "test_dataset_005.yaml"),
        ]

        self.split_classes = [
            SplitDataSetA,
            SplitDataSetAll,
        ]

        self.target_features_temps = [128, 132342]
        self.target_features_psals = [140, 132342]

        self.test_data_file = (
            Path(__file__).resolve().parent
            / "data"
            / "input"
            / "nrt_cora_bo_test.parquet"
        )

    def _set_config(self, idx):
        self.config = DataSetConfig(str(self.config_file_paths[idx]))
        self.config.select("NRT_BO_001")

    @pytest.mark.parametrize("idx", range(2))
    def test_load_dataset_valid_config(self, idx):
        """
        Check that load_step6_split_dataset returns a SplitDataSetA instance
        with the correct step name.
        """
        self._set_config(idx)
        ds = load_step6_split_dataset(self.config)
        assert isinstance(ds, self.split_classes[idx])
        assert ds.step_name == "split"

    @pytest.mark.parametrize("idx", range(2))
    def test_load_dataset_input_data(self, idx):
        """
        Check that load_step6_split_dataset properly sets the `target_features`
        input provided from previous steps and retains its expected structure.
        """
        self._set_config(idx)
        ds_input = load_step1_input_dataset(self.config)
        ds_input.input_file_name = str(self.test_data_file)
        ds_input.read_input_data()

        ds_select = load_step3_select_dataset(self.config, ds_input.input_data)
        ds_select.label_profiles()

        ds_summary = load_step2_summary_dataset(self.config, ds_input.input_data)
        ds_summary.calculate_stats()

        ds_locate = load_step4_locate_dataset(
            self.config, ds_input.input_data, ds_select.selected_profiles
        )
        ds_locate.process_targets()

        ds_extract = load_step5_extract_dataset(
            self.config,
            ds_input.input_data,
            ds_select.selected_profiles,
            ds_locate.selected_rows,
            ds_summary.summary_stats,
        )
        ds_extract.process_targets()

        ds = load_step6_split_dataset(self.config, ds_extract.target_features)

        assert isinstance(ds, self.split_classes[idx])

        assert isinstance(ds.target_features["temp"], pl.DataFrame)
        assert ds.target_features["temp"].shape[0] == self.target_features_temps[idx]
        assert ds.target_features["temp"].shape[1] == 58

        assert isinstance(ds.target_features["psal"], pl.DataFrame)
        assert ds.target_features["psal"].shape[0] == self.target_features_psals[idx]
        assert ds.target_features["psal"].shape[1] == 58
