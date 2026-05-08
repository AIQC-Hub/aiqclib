"""Unit tests for the InputDataSetA class.
This module verifies the functionality of InputDataSetA, including
reading data from various file types, resolving input file paths,
applying column renames, and filtering rows based on year criteria.
"""

import unittest
from pathlib import Path

import polars as pl

from aiqclib.common.config.dataset_config import DataSetConfig
from aiqclib.prepare.step1_read_input.dataset_a import InputDataSetA


class TestInputDataSetA(unittest.TestCase):
    """
    Tests for verifying input data reading and resolution in the InputDataSetA class.
    Ensures data is loaded as expected from Parquet files, file names are resolved
    properly, and property checks are correct.
    """

    def setUp(self):
        """
        Set up the test configuration objects and specify test data file paths.
        """
        self.config_file_path = (
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

        self.config_file_path2 = (
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_dataset_002.yaml"
        )
        self.config2 = DataSetConfig(str(self.config_file_path2))
        self.config2.select("NRT_BO_001")

    def _get_input_data(self, config, file_type=None, read_file_options=None):
        """
        Helper method that loads input data into a Polars DataFrame, optionally
        setting the file type and read options in the config before reading.
        """
        ds = InputDataSetA(config)
        ds.input_file_name = str(self.test_data_file)

        if file_type is not None:
            ds.config.data["step_param_set"]["steps"]["input"]["file_type"] = file_type

        if read_file_options is not None:
            ds.config.data["step_param_set"]["steps"]["input"]["read_file_options"] = (
                read_file_options
            )

        ds.read_input_data()
        return ds.input_data

    def test_step_name(self):
        """
        Confirm that InputDataSetA instances declare their step name as "input".
        """
        ds = InputDataSetA(self.config)
        self.assertEqual(ds.step_name, "input")

    def test_input_file_name(self):
        """
        Ensure the input file name is determined correctly based on
        the loaded configuration.
        """
        ds = InputDataSetA(self.config)
        self.assertEqual(
            "/path/to/input_1/input_folder_1/nrt_cora_bo_test.parquet",
            str(ds.input_file_name),
        )

    def test_read_input_data_with_explicit_type(self):
        """
        Verify that data is read from a Parquet file when a 'parquet' file_type
        is explicitly specified in the config.
        """
        df = self._get_input_data(
            self.config, file_type="parquet", read_file_options={}
        )
        self.assertIsInstance(df, pl.DataFrame)
        self.assertEqual(df.shape[0], 132342)
        self.assertEqual(df.shape[1], 30)

    def test_read_input_data_infer_type(self):
        """
        Verify that data reading automatically infers the file type
        (from extension) when file_type is not explicitly set.
        """
        df = self._get_input_data(self.config, file_type=None, read_file_options={})
        self.assertIsInstance(df, pl.DataFrame)
        self.assertEqual(df.shape[0], 132342)
        self.assertEqual(df.shape[1], 30)

    def test_read_input_data_missing_options(self):
        """
        Confirm that data can be read successfully when no additional
        file reading options are provided.
        """
        df = self._get_input_data(
            self.config, file_type="parquet", read_file_options=None
        )
        self.assertIsInstance(df, pl.DataFrame)
        self.assertEqual(df.shape[0], 132342)
        self.assertEqual(df.shape[1], 30)

    def test_read_input_data_file_not_found(self):
        """
        Ensure that attempting to read a non-existent input file
        raises a FileNotFoundError.
        """
        ds = InputDataSetA(self.config)
        ds.input_file_name = str(self.test_data_file) + "_not_found"

        with self.assertRaises(FileNotFoundError):
            ds.read_input_data()

    def test_read_input_data_with_extra_options(self):
        """
        Verify that additional reading options (e.g., read only 100 rows)
        can be passed and applied to the data loading process.
        """
        df = self._get_input_data(
            self.config, file_type="parquet", read_file_options={"n_rows": 100}
        )
        self.assertIsInstance(df, pl.DataFrame)
        self.assertEqual(df.shape[0], 100)
        self.assertEqual(df.shape[1], 30)


class TestInputDataSetARename(unittest.TestCase):
    """
    Tests for verifying that InputDataSetA applies a rename dictionary
    to columns when specified in the configuration.
    """

    def setUp(self):
        """
        Load a separate test configuration and point to a known Parquet file.
        """
        self.config_file_path = (
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_dataset_002.yaml"
        )
        self.config = DataSetConfig(str(self.config_file_path))
        self.config.select("NRT_BO_001")
        self.test_data_file = (
            Path(__file__).resolve().parent
            / "data"
            / "input"
            / "nrt_cora_bo_test.parquet"
        )

    def _get_input_data(self, config, file_type=None, read_file_options=None):
        """
        Helper method that loads input data into a Polars DataFrame, optionally
        setting the file type and read options in the config before reading.
        """
        ds = InputDataSetA(config)
        ds.input_file_name = str(self.test_data_file)

        if file_type is not None:
            ds.config.data["step_param_set"]["steps"]["input"]["file_type"] = file_type

        if read_file_options is not None:
            ds.config.data["step_param_set"]["steps"]["input"]["read_file_options"] = (
                read_file_options
            )

        ds.read_input_data()
        return ds.input_data

    def test_rename(self):
        """
        Ensure columns are correctly renamed (e.g., 'filename' -> 'filename_new'),
        as specified by rename_dict in the configuration.
        """
        df = self._get_input_data(
            self.config, file_type="parquet", read_file_options={}
        )
        self.assertFalse("filename" in df.columns)
        self.assertTrue("filename_new" in df.columns)

    def test_rename_with_incorrect_param(self):
        """
        Validate that if rename_dict is removed from config parameters,
        original column names are retained.
        """
        del self.config.get_step_params("input")["rename_dict"]
        df = self._get_input_data(
            self.config, file_type="parquet", read_file_options={}
        )
        self.assertTrue("filename" in df.columns)


class TestInputDataSetAFilter(unittest.TestCase):
    """
    Tests for verifying the behavior of filtering years in the InputDataSetA class,
    ensuring that filter_rows options in the config are correctly applied
    to prune or retain specific years.
    """

    def setUp(self):
        """
        Load a test configuration for dataset_002, define the input file path,
        and prepare for data filtering tests.
        """
        self.config_file_path = (
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_dataset_002.yaml"
        )
        self.config = DataSetConfig(str(self.config_file_path))
        self.config.select("NRT_BO_001")
        self.test_data_file = (
            Path(__file__).resolve().parent
            / "data"
            / "input"
            / "nrt_cora_bo_test.parquet"
        )

    def _get_input_data(self, config, file_type=None, read_file_options=None):
        """
        Helper method that loads input data into a Polars DataFrame, optionally
        setting the file type and read options in the config before reading.
        """
        ds = InputDataSetA(config)
        ds.input_file_name = str(self.test_data_file)

        if file_type is not None:
            ds.config.data["step_param_set"]["steps"]["input"]["file_type"] = file_type

        if read_file_options is not None:
            ds.config.data["step_param_set"]["steps"]["input"]["read_file_options"] = (
                read_file_options
            )

        ds.read_input_data()
        return ds.input_data

    @staticmethod
    def _get_uniq_years(df):
        """
        Internal helper to extract the unique years from the 'profile_timestamp' column.
        """
        return (
            df.select(pl.col("profile_timestamp").dt.year().unique())
            .to_series()
            .to_list()
        )

    def test_remove_years_without_filter_rows_flag(self):
        """
        Ensure that when filter_rows is disabled, no years are removed
        and all available years remain in the DataFrame.
        """
        self.config.get_step_params("input")["sub_steps"]["filter_rows"] = False
        df = self._get_input_data(
            self.config, file_type="parquet", read_file_options={}
        )
        self.assertEqual(
            self._get_uniq_years(df),
            [2017, 2018, 2019, 2020, 2021, 2022, 2023],
        )

    def test_remove_years_with_empty_array(self):
        """
        Confirm that specifying an empty remove_years and keep_years list implies
        no specific filtering takes place, retaining all years.
        """
        self.config.get_step_params("input")["sub_steps"]["filter_rows"] = True
        self.config.get_step_params("input")["filter_method_dict"]["remove_years"] = []
        self.config.get_step_params("input")["filter_method_dict"]["keep_years"] = []

        df = self._get_input_data(
            self.config, file_type="parquet", read_file_options={}
        )
        self.assertEqual(
            self._get_uniq_years(df),
            [2017, 2018, 2019, 2020, 2021, 2022, 2023],
        )

    def test_remove_years(self):
        """
        Check that specifying years in remove_years excludes those years
        from the resulting DataFrame.
        """
        self.config.get_step_params("input")["sub_steps"]["filter_rows"] = True
        self.config.get_step_params("input")["filter_method_dict"]["remove_years"] = [
            2022,
            2023,
        ]
        self.config.get_step_params("input")["filter_method_dict"]["keep_years"] = []

        df = self._get_input_data(
            self.config, file_type="parquet", read_file_options={}
        )
        self.assertEqual(
            self._get_uniq_years(df),
            [2017, 2018, 2019, 2020, 2021],
        )

    def test_keep_years(self):
        """
        Check that specifying years in keep_years retains only those years
        from the resulting DataFrame.
        """
        self.config.get_step_params("input")["sub_steps"]["filter_rows"] = True
        self.config.get_step_params("input")["filter_method_dict"]["remove_years"] = []
        self.config.get_step_params("input")["filter_method_dict"]["keep_years"] = [
            2022,
            2023,
        ]

        df = self._get_input_data(
            self.config, file_type="parquet", read_file_options={}
        )
        self.assertEqual(
            self._get_uniq_years(df),
            [2022, 2023],
        )
