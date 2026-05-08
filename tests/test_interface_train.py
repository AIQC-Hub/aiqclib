"""
This module contains unit tests for the training and evaluation pipeline.
It verifies that the `train_and_evaluate` function correctly generates
expected output files and directories, including validation reports and
trained model artifacts.
"""

import os
import shutil
import unittest
import pytest
from pathlib import Path

from aiqclib.common.config.training_config import TrainingConfig
from aiqclib.interface.train import train_and_evaluate


class TestCreateTrainingDataSet:
    """
    A suite of tests ensuring that training and evaluation steps
    generate the expected output files and folders.
    """

    def _setup_configs(self):
        self.configs = []
        for x in self.config_file_paths:
            config = TrainingConfig(str(x))
            config.select("NRT_BO_001")
            config.data["path_info"] = {
                "name": "data_set_1",
                "common": {"base_path": str(self.test_data_location)},
                "input": {
                    "base_path": str(self.input_data_path),
                    "step_folder_name": "..",
                },
            }
            self.configs.append(config)

    @pytest.fixture(autouse=True)
    def setup_and_clean(self):
        """
        Prepare the test environment by loading a specified training configuration,
        and defining input/output paths for subsequent training/evaluation tests.
        """
        dir_config = Path(__file__).resolve().parent / "data" / "config"
        self.config_file_paths = [
            dir_config / "test_training_001.yaml",
            dir_config / "test_training_002.yaml",
        ]
        self.test_data_location = Path(__file__).resolve().parent / "data" / "test"
        self.input_data_path = str(
            Path(__file__).resolve().parent / "data" / "training"
        )
        self._setup_configs()

        yield

        for x in self.configs:
            output_folder = self.test_data_location / x.data["dataset_folder_name"]
            if output_folder.exists() and output_folder.is_dir():
                shutil.rmtree(output_folder)

    @pytest.mark.parametrize("idx", range(2))
    def test_train_and_evaluate(self, idx):
        """
        Check that train_and_evaluate runs end-to-end and produces
        validation results and trained model artifacts.
        """
        # Execute the training and evaluation process
        train_and_evaluate(self.configs[idx])

        # Define the expected output folder based on the configuration
        output_folder = (
            self.test_data_location / self.configs[idx].data["dataset_folder_name"]
        )
        dir_validate = output_folder / "validate"
        dir_build = output_folder / "build"
        dir_model = output_folder / "model"

        # Assert that expected validation report files are created
        assert os.path.exists(dir_validate / "validation_report_temp.tsv")
        assert os.path.exists(dir_validate / "validation_report_psal.tsv")
        assert os.path.exists(dir_validate / "validation_report_pres.tsv")

        # Assert that expected contingency table files are created
        assert os.path.exists(dir_validate / "contingency_tables_temp.parquet")
        assert os.path.exists(dir_validate / "contingency_tables_psal.parquet")
        assert os.path.exists(dir_validate / "contingency_tables_pres.parquet")

        # Assert that expected metric plot files are created
        assert os.path.exists(dir_validate / "metric_plots_temp.svg")
        assert os.path.exists(dir_validate / "metric_plots_psal.svg")
        assert os.path.exists(dir_validate / "metric_plots_pres.svg")

        # Assert that expected build report files are created
        assert os.path.exists(dir_build / "test_report_temp.tsv")
        assert os.path.exists(dir_build / "test_report_psal.tsv")
        assert os.path.exists(dir_build / "test_report_pres.tsv")

        # Assert that expected contingency table files are created
        assert os.path.exists(dir_build / "test_contingency_tables_temp.parquet")
        assert os.path.exists(dir_build / "test_contingency_tables_psal.parquet")
        assert os.path.exists(dir_build / "test_contingency_tables_pres.parquet")

        # Assert that expected metric plot files are created
        assert os.path.exists(dir_build / "test_metric_plots_temp.svg")
        assert os.path.exists(dir_build / "test_metric_plots_psal.svg")
        assert os.path.exists(dir_build / "test_metric_plots_pres.svg")

        # Assert that expected SHAP files are NOT created
        assert not os.path.exists(dir_build / "test_shap_values_temp.parquet")
        assert not os.path.exists(dir_build / "test_shap_values_psal.parquet")
        assert not os.path.exists(dir_build / "test_shap_values_pres.parquet")

        # Assert that expected trained model files are created
        assert os.path.exists(dir_model / "model_temp.joblib")
        assert os.path.exists(dir_model / "model_psal.joblib")
        assert os.path.exists(dir_model / "model_pres.joblib")

    @pytest.mark.parametrize("idx", range(2))
    def test_shap_value_output(self, idx):
        # Execute the training and evaluation process with SHAP output
        self.configs[idx].data["step_param_set"]["steps"]["model"]["calculate_shap"] = (
            True
        )
        train_and_evaluate(self.configs[idx])

        # Define the expected output folder based on the configuration
        output_folder = (
            self.test_data_location / self.configs[idx].data["dataset_folder_name"]
        )
        dir_build = output_folder / "build"

        # Assert that expected SHAP files are created
        assert os.path.exists(dir_build / "test_shap_values_temp.parquet")
        assert os.path.exists(dir_build / "test_shap_values_psal.parquet")
        assert os.path.exists(dir_build / "test_shap_values_pres.parquet")


class TestCreateTrainingDataSetNegX5(unittest.TestCase):
    """
    A suite of tests ensuring that training and evaluation steps
    generate the expected output files and folders.
    """

    def setUp(self):
        """
        Prepare the test environment by loading a specified training configuration,
        and defining input/output paths for subsequent training/evaluation tests.
        """
        self.config_file_path = str(
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_training_003.yaml"
        )
        self.config = TrainingConfig(str(self.config_file_path))
        self.config.select("NRT_BO_001")
        self.test_data_location = Path(__file__).resolve().parent / "data" / "test"
        self.input_data_path = (
            Path(__file__).resolve().parent / "data" / "negx5_training"
        )

        # Configure the dataset paths for the test
        self.config.data["path_info"] = {
            "name": "data_set_1",
            "common": {"base_path": str(self.test_data_location)},
            "input": {
                "base_path": str(self.input_data_path),
                "step_folder_name": "..",
            },
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

    def test_train_and_evaluate(self):
        """
        Check that train_and_evaluate runs end-to-end and produces
        validation results and trained model artifacts.
        """
        # Execute the training and evaluation process
        train_and_evaluate(self.config)

        # Define the expected output folder based on the configuration
        output_folder = (
            self.test_data_location / self.config.data["dataset_folder_name"]
        )
        dir_validate = output_folder / "validate"
        dir_build = output_folder / "build"
        dir_model = output_folder / "model"

        # Assert that expected validation report files are created
        self.assertTrue(os.path.exists(dir_validate / "validation_report_temp.tsv"))
        self.assertTrue(os.path.exists(dir_validate / "validation_report_psal.tsv"))
        self.assertTrue(os.path.exists(dir_validate / "validation_report_pres.tsv"))

        # Assert that expected contingency table files are created
        self.assertTrue(
            os.path.exists(dir_validate / "contingency_tables_temp.parquet")
        )
        self.assertTrue(
            os.path.exists(dir_validate / "contingency_tables_psal.parquet")
        )
        self.assertTrue(
            os.path.exists(dir_validate / "contingency_tables_pres.parquet")
        )

        # Assert that expected metric plot files are created
        self.assertTrue(os.path.exists(dir_validate / "metric_plots_temp.svg"))
        self.assertTrue(os.path.exists(dir_validate / "metric_plots_psal.svg"))
        self.assertTrue(os.path.exists(dir_validate / "metric_plots_pres.svg"))

        # Assert that expected build report files are created
        self.assertTrue(os.path.exists(dir_build / "test_report_temp.tsv"))
        self.assertTrue(os.path.exists(dir_build / "test_report_psal.tsv"))
        self.assertTrue(os.path.exists(dir_build / "test_report_pres.tsv"))

        # Assert that expected contingency table files are created
        self.assertTrue(
            os.path.exists(dir_build / "test_contingency_tables_temp.parquet")
        )
        self.assertTrue(
            os.path.exists(dir_build / "test_contingency_tables_psal.parquet")
        )
        self.assertTrue(
            os.path.exists(dir_build / "test_contingency_tables_pres.parquet")
        )

        # Assert that expected metric plot files are created
        self.assertTrue(os.path.exists(dir_build / "test_metric_plots_temp.svg"))
        self.assertTrue(os.path.exists(dir_build / "test_metric_plots_psal.svg"))
        self.assertTrue(os.path.exists(dir_build / "test_metric_plots_pres.svg"))

        # Assert that expected trained model files are created
        self.assertTrue(os.path.exists(dir_model / "model_temp.joblib"))
        self.assertTrue(os.path.exists(dir_model / "model_psal.joblib"))
        self.assertTrue(os.path.exists(dir_model / "model_pres.joblib"))
