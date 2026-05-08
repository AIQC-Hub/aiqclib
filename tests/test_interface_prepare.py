"""
Unit tests for the `create_training_dataset` function, verifying its ability to
generate the correct directory structure and output files for various data
processing steps within a training dataset.
"""

import os
import shutil
import unittest
import pytest
from pathlib import Path

from aiqclib.common.config.dataset_config import DataSetConfig
from aiqclib.interface.prepare import create_training_dataset


class TestCreateTrainingDataSet:
    """
    Tests for verifying that create_training_dataset produces the
    expected directory structure and output files for training data.
    """

    def _setup_configs(self):
        self.configs = []
        for x in self.config_file_paths:
            c = DataSetConfig(str(x))
            c.select("NRT_BO_001")
            c.data["input_file_name"] = "nrt_cora_bo_test.parquet"
            c.data["path_info"] = {
                "name": "data_set_1",
                "common": {"base_path": str(self.test_data_location)},
                "input": {
                    "base_path": str(self.input_data_path),
                    "step_folder_name": "",
                },
            }
            self.configs.append(c)

    @pytest.fixture(autouse=True)
    def setup_and_cleanup(self):
        """
        Prepare the test environment by creating a DataSetConfig object,
        defining file paths, and updating the configuration with test input
        and output paths.
        """
        dir_config = Path(__file__).resolve().parent / "data" / "config"
        self.config_file_paths = [
            (dir_config / "test_dataset_001.yaml"),
            (dir_config / "test_dataset_004.yaml"),
            (dir_config / "test_dataset_005.yaml"),
        ]
        self.test_data_location = Path(__file__).resolve().parent / "data" / "test"
        self.input_data_path = Path(__file__).resolve().parent / "data" / "input"

        self._setup_configs()
        yield

        for c in self.configs:
            output_folder = self.test_data_location / c.data["dataset_folder_name"]
            if os.path.exists(output_folder):
                shutil.rmtree(output_folder)

    @pytest.mark.parametrize("idx", range(3))
    def test_create_training_data_set(self, idx):
        """
        Verify that `create_training_dataset` generates the expected folder
        hierarchy and all required output files for summary, select, locate,
        extract, and split steps.
        """
        create_training_dataset(self.configs[idx])

        output_folder = (
            self.test_data_location / self.configs[idx].data["dataset_folder_name"]
        )
        dir_summary = output_folder / "summary"
        dir_select = output_folder / "select"
        dir_locate = output_folder / "locate"
        dir_extract = output_folder / "extract"
        dir_split = output_folder / "split"
        assert os.path.exists(str(dir_summary / "summary_stats.tsv"))
        assert os.path.exists(str(dir_select / "selected_profiles.parquet"))
        assert os.path.exists(str(dir_locate / "selected_rows_temp.parquet"))
        assert os.path.exists(str(dir_locate / "selected_rows_psal.parquet"))
        assert os.path.exists(str(dir_locate / "selected_rows_pres.parquet"))
        assert os.path.exists(str(dir_extract / "extracted_features_temp.parquet"))
        assert os.path.exists(str(dir_extract / "extracted_features_psal.parquet"))
        assert os.path.exists(str(dir_extract / "extracted_features_pres.parquet"))
        assert os.path.exists(str(dir_split / "train_set_temp.parquet"))
        assert os.path.exists(str(dir_split / "train_set_psal.parquet"))
        assert os.path.exists(str(dir_split / "train_set_pres.parquet"))
        assert os.path.exists(str(dir_split / "test_set_temp.parquet"))
        assert os.path.exists(str(dir_split / "test_set_psal.parquet"))
        assert os.path.exists(str(dir_split / "test_set_pres.parquet"))


class TestCreateTrainingDataSetNegX5(unittest.TestCase):
    """
    Tests for verifying that create_training_dataset produces the
    expected directory structure and output files for training data when
    custom step folder names are configured.
    """

    def setUp(self):
        """
        Prepare the test environment by creating a DataSetConfig object,
        defining file paths, and updating the configuration with test input
        and output paths, including a custom folder name for the 'split' step.
        """
        self.config_file_path = str(
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_dataset_003.yaml"
        )
        self.config = DataSetConfig(str(self.config_file_path))
        self.config.select("NRT_BO_001")
        self.config.data["input_file_name"] = "nrt_cora_bo_test.parquet"
        self.test_data_location = Path(__file__).resolve().parent / "data" / "test"
        self.input_data_path = Path(__file__).resolve().parent / "data" / "input"
        self.config.data["path_info"] = {
            "name": "data_set_1",
            "common": {"base_path": str(self.test_data_location)},
            "input": {"base_path": str(self.input_data_path), "step_folder_name": ""},
            "split": {"step_folder_name": "training"},
        }

    def test_create_training_data_set(self):
        """
        Verify that `create_training_dataset` generates the expected folder
        hierarchy and all required output files, specifically checking the
        custom 'training' folder for split data.
        """
        create_training_dataset(self.config)

        output_folder = (
            self.test_data_location / self.config.data["dataset_folder_name"]
        )

        self.assertTrue(
            os.path.exists(str(output_folder / "summary" / "summary_stats.tsv"))
        )
        self.assertTrue(
            os.path.exists(str(output_folder / "select" / "selected_profiles.parquet"))
        )
        self.assertTrue(
            os.path.exists(str(output_folder / "locate" / "selected_rows_temp.parquet"))
        )
        self.assertTrue(
            os.path.exists(str(output_folder / "locate" / "selected_rows_psal.parquet"))
        )
        self.assertTrue(
            os.path.exists(str(output_folder / "locate" / "selected_rows_pres.parquet"))
        )
        self.assertTrue(
            os.path.exists(
                str(output_folder / "extract" / "extracted_features_temp.parquet")
            )
        )
        self.assertTrue(
            os.path.exists(
                str(output_folder / "extract" / "extracted_features_psal.parquet")
            )
        )
        self.assertTrue(
            os.path.exists(
                str(output_folder / "extract" / "extracted_features_pres.parquet")
            )
        )
        # Verify files are in the custom 'training' folder instead of 'split'
        self.assertTrue(
            os.path.exists(str(output_folder / "training" / "train_set_temp.parquet"))
        )
        self.assertTrue(
            os.path.exists(str(output_folder / "training" / "train_set_psal.parquet"))
        )
        self.assertTrue(
            os.path.exists(str(output_folder / "training" / "train_set_pres.parquet"))
        )
        self.assertTrue(
            os.path.exists(str(output_folder / "training" / "test_set_temp.parquet"))
        )
        self.assertTrue(
            os.path.exists(str(output_folder / "training" / "test_set_psal.parquet"))
        )
        self.assertTrue(
            os.path.exists(str(output_folder / "training" / "test_set_pres.parquet"))
        )

    def tearDown(self):
        """
        Clean up the test environment by removing the generated output directory
        and its contents after each test.
        """
        output_folder = (
            self.test_data_location / self.config.data["dataset_folder_name"]
        )
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
