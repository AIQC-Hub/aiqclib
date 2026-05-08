"""
Unit tests for the `classify_dataset` function.

This module contains tests to verify that the `classify_dataset`
function correctly processes input data and generates the expected
directory structure and output files for various classification steps,
including summary, selection, location, extraction, classification,
and concatenation.
"""

import os
import shutil
import unittest
import pytest
from pathlib import Path

from aiqclib.common.config.classify_config import ClassificationConfig
from aiqclib.interface.classify import classify_dataset


class TestClassifyDataSet:
    """
    Tests for verifying that classify_dataset produces the
    expected directory structure and output files for classification processes.
    """

    def _setup_configs(self):
        self.configs = []
        for x in self.config_file_paths:
            c = ClassificationConfig(str(x))
            c.select("NRT_BO_001")
            c.data["input_file_name"] = "nrt_cora_bo_test.parquet"
            c.data["path_info"] = {
                "name": "data_set_1",
                "common": {"base_path": str(self.test_data_location)},
                "input": {
                    "base_path": str(self.input_data_path),
                    "step_folder_name": "",
                },
                "model": {
                    "base_path": str(self.data_path),
                    "step_folder_name": "training",
                },
                "concat": {"step_folder_name": "classify"},
            }
            self.configs.append(c)

    @pytest.fixture(autouse=True)
    def setup_and_cleanup(self):
        """
        Prepare the test environment by creating a DataSetConfig object,
        defining file paths, and updating the configuration with test input
        and output paths. This method ensures that each test starts with a
        clean and predictable configuration.
        """
        config_path = Path(__file__).resolve().parent / "data" / "config"
        self.config_file_paths = [
            config_path / "test_classify_001.yaml",
            config_path / "test_classify_002.yaml",
        ]
        self.test_data_location = Path(__file__).resolve().parent / "data" / "test"
        self.data_path = Path(__file__).resolve().parent / "data"
        self.input_data_path = self.data_path / "input"

        self._setup_configs()
        yield

        for c in self.configs:
            output_folder = self.test_data_location / c.data["dataset_folder_name"]
            if output_folder.exists() and output_folder.is_dir():
                shutil.rmtree(output_folder)

    @pytest.mark.parametrize("idx", range(2))
    def test_classify_data_set(self, idx):
        """
        Verifies that the `classify_dataset` function generates the expected
        folder hierarchy and files for each step of the classification process.
        This includes checking for the presence of summary statistics, selected profiles,
        located rows, extracted features, classification predictions, and reports,
        as well as the final concatenated predictions.
        """
        classify_dataset(self.configs[idx])

        output_folder = (
            self.test_data_location / self.configs[idx].data["dataset_folder_name"]
        )
        dir_summary = output_folder / "summary"
        dir_select = output_folder / "select"
        dir_locate = output_folder / "locate"
        dir_extract = output_folder / "extract"
        dir_classify = output_folder / "classify"

        assert os.path.exists(str(dir_summary / "summary_stats_classify.tsv"))
        assert os.path.exists(str(dir_select / "selected_profiles_classify.parquet"))
        assert os.path.exists(str(dir_locate / "selected_rows_classify_temp.parquet"))
        assert os.path.exists(str(dir_locate / "selected_rows_classify_psal.parquet"))
        assert os.path.exists(str(dir_locate / "selected_rows_classify_pres.parquet"))
        assert os.path.exists(
            str(dir_extract / "extracted_features_classify_temp.parquet")
        )
        assert os.path.exists(
            str(dir_extract / "extracted_features_classify_psal.parquet")
        )
        assert os.path.exists(
            str(dir_extract / "extracted_features_classify_pres.parquet")
        )
        assert os.path.exists(str(dir_classify / "classify_prediction_temp.parquet"))
        assert os.path.exists(str(dir_classify / "classify_prediction_psal.parquet"))
        assert os.path.exists(str(dir_classify / "classify_prediction_pres.parquet"))
        assert os.path.exists(str(dir_classify / "classify_report_temp.tsv"))
        assert os.path.exists(str(dir_classify / "classify_report_psal.tsv"))
        assert os.path.exists(str(dir_classify / "classify_report_pres.tsv"))
        assert os.path.exists(
            str(dir_classify / "classify_contingency_tables_temp.parquet")
        )
        assert os.path.exists(
            str(dir_classify / "classify_contingency_tables_psal.parquet")
        )
        assert os.path.exists(
            str(dir_classify / "classify_contingency_tables_pres.parquet")
        )
        assert os.path.exists(str(dir_classify / "classify_metric_plots_temp.svg"))
        assert os.path.exists(str(dir_classify / "classify_metric_plots_psal.svg"))
        assert os.path.exists(str(dir_classify / "classify_metric_plots_pres.svg"))

        # Assert that expected SHAP files are NOT created
        assert not os.path.exists(dir_classify / "classify_shap_values_temp.parquet")
        assert not os.path.exists(dir_classify / "classify_shap_values_psal.parquet")
        assert not os.path.exists(dir_classify / "classify_shap_values_pres.parquet")

        assert os.path.exists(str(dir_classify / "predictions.parquet"))

    @pytest.mark.parametrize("idx", range(2))
    def test_shap_value_output(self, idx):
        """
        Verifies that the `classify_dataset` function generates the expected
        SHAP output files
        """
        self.configs[idx].data["step_param_set"]["steps"]["model"]["calculate_shap"] = (
            True
        )
        classify_dataset(self.configs[idx])

        output_folder = (
            self.test_data_location / self.configs[idx].data["dataset_folder_name"]
        )
        dir_classify = output_folder / "classify"

        # Assert that expected SHAP files are created
        assert os.path.exists(dir_classify / "classify_shap_values_temp.parquet")
        assert os.path.exists(dir_classify / "classify_shap_values_psal.parquet")
        assert os.path.exists(dir_classify / "classify_shap_values_pres.parquet")


class TestClassifyDataSetNegX5(unittest.TestCase):
    """
    Tests for verifying that classify_dataset produces the
    expected directory structure and output files for classification processes
    when using a different model path (e.g., for 'negx5_model').
    """

    def setUp(self):
        """
        Prepare the test environment by creating a DataSetConfig object,
        defining file paths, and updating the configuration with test input
        and output paths. This method ensures that each test starts with a
        clean and predictable configuration.
        """
        self.config_file_path = str(
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_classify_001.yaml"
        )
        self.config = ClassificationConfig(str(self.config_file_path))
        self.config.select("NRT_BO_001")
        self.config.data["input_file_name"] = "nrt_cora_bo_test.parquet"
        self.test_data_location = Path(__file__).resolve().parent / "data" / "test"
        self.data_path = Path(__file__).resolve().parent / "data"
        self.input_data_path = self.data_path / "input"
        self.config.data["path_info"] = {
            "name": "data_set_1",
            "common": {"base_path": str(self.test_data_location)},
            "input": {"base_path": str(self.input_data_path), "step_folder_name": ""},
            "model": {
                "base_path": str(self.data_path),
                "step_folder_name": "negx5_model",
            },
            "concat": {"step_folder_name": "classify"},
        }

    def tearDown(self):
        """
        Clean up the test environment by removing any generated output folders
        and files after each test method has completed.
        """
        output_folder = (
            self.test_data_location / self.config.data["dataset_folder_name"]
        )
        if output_folder.exists() and output_folder.is_dir():
            shutil.rmtree(output_folder)

    def test_classify_data_set(self):
        """
        Verifies that the `classify_dataset` function generates the expected
        folder hierarchy and files for each step of the classification process.
        This includes checking for the presence of summary statistics, selected profiles,
        located rows, extracted features, classification predictions, and reports,
        as well as the final concatenated predictions.
        """
        classify_dataset(self.config)

        output_folder = (
            self.test_data_location / self.config.data["dataset_folder_name"]
        )
        dir_summary = output_folder / "summary"
        dir_select = output_folder / "select"
        dir_locate = output_folder / "locate"
        dir_extract = output_folder / "extract"
        dir_classify = output_folder / "classify"

        self.assertTrue(os.path.exists(str(dir_summary / "summary_stats_classify.tsv")))
        self.assertTrue(
            os.path.exists(str(dir_select / "selected_profiles_classify.parquet"))
        )
        self.assertTrue(
            os.path.exists(str(dir_locate / "selected_rows_classify_temp.parquet"))
        )
        self.assertTrue(
            os.path.exists(str(dir_locate / "selected_rows_classify_psal.parquet"))
        )
        self.assertTrue(
            os.path.exists(str(dir_locate / "selected_rows_classify_pres.parquet"))
        )
        self.assertTrue(
            os.path.exists(
                str(dir_extract / "extracted_features_classify_temp.parquet")
            )
        )
        self.assertTrue(
            os.path.exists(
                str(dir_extract / "extracted_features_classify_psal.parquet")
            )
        )
        self.assertTrue(
            os.path.exists(
                str(dir_extract / "extracted_features_classify_pres.parquet")
            )
        )
        self.assertTrue(
            os.path.exists(str(dir_classify / "classify_prediction_temp.parquet"))
        )
        self.assertTrue(
            os.path.exists(str(dir_classify / "classify_prediction_psal.parquet"))
        )
        self.assertTrue(
            os.path.exists(str(dir_classify / "classify_prediction_pres.parquet"))
        )
        self.assertTrue(os.path.exists(str(dir_classify / "classify_report_temp.tsv")))
        self.assertTrue(os.path.exists(str(dir_classify / "classify_report_psal.tsv")))
        self.assertTrue(os.path.exists(str(dir_classify / "classify_report_pres.tsv")))
        self.assertTrue(
            os.path.exists(
                str(dir_classify / "classify_contingency_tables_temp.parquet")
            )
        )
        self.assertTrue(
            os.path.exists(
                str(dir_classify / "classify_contingency_tables_psal.parquet")
            )
        )
        self.assertTrue(
            os.path.exists(
                str(dir_classify / "classify_contingency_tables_pres.parquet")
            )
        )
        self.assertTrue(
            os.path.exists(str(dir_classify / "classify_metric_plots_temp.svg"))
        )
        self.assertTrue(
            os.path.exists(str(dir_classify / "classify_metric_plots_psal.svg"))
        )
        self.assertTrue(
            os.path.exists(str(dir_classify / "classify_metric_plots_pres.svg"))
        )

        self.assertTrue(os.path.exists(str(dir_classify / "predictions.parquet")))
