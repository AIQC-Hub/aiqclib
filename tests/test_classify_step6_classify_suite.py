"""
This module contains unit tests for the `ClassifyAllSuite` class,
focusing on its functionality for loading multiple models, testing them
against input data, and saving aggregated classification reports,
contingency tables, and predictions.
"""

import os
import pytest
from pathlib import Path

import matplotlib
import polars as pl

# Use non-interactive backend to prevent plots from trying to open windows during tests
matplotlib.use("Agg")

from aiqclib.classify.step6_classify_dataset.dataset_all_suite import ClassifyAllSuite
from aiqclib.common.config.classify_config import ClassificationConfig
from aiqclib.train.models.model_suite import ModelSuite
from aiqclib.common.loader.classify_loader import (
    load_classify_step1_input_dataset,
    load_classify_step2_summary_dataset,
    load_classify_step3_select_dataset,
    load_classify_step4_locate_dataset,
    load_classify_step5_extract_dataset,
)

TEST_COUNT = 1


class TestClassifyAllSuiteClass:
    """
    A suite of tests ensuring that the `ClassifyAllSuite` step correctly
    initializes paths and base models before execution.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test environment and basic configuration."""
        config_path = Path(__file__).resolve().parent / "data" / "config"
        self.config_file = config_path / "test_classify_001.yaml"
        self.config = ClassificationConfig(self.config_file)
        self.config.select("NRT_BO_001")

        # Force config to use ModelSuite and limit methods to keep tests fast
        self.config.data["step_class_set"]["steps"]["classify"] = "ClassifyAllSuite"
        self.config.data["step_class_set"]["steps"]["model"] = "ModelSuite"
        self.config.data["step_param_set"]["steps"]["model"] = {
            "methods": ["XGB", "DT"]
        }

    def test_step_name(self):
        """Check that the ClassifyAllSuite step name is correctly assigned."""
        ds = ClassifyAllSuite(self.config)
        assert ds.step_name == "classify"

    def test_multi_flag_check(self):
        """
        Verify that instantiating ClassifyAllSuite with a standard model
        (multi=False) raises a ValueError.
        """
        self.config.data["step_class_set"]["steps"]["model"] = "XGBoost"
        with pytest.raises(ValueError, match="multi=True"):
            _ = ClassifyAllSuite(self.config)

    def test_output_file_names(self):
        """
        Verify that default output file names correctly reflect composite keys
        for models, and aggregated target keys for reports/predictions.
        """
        ds = ClassifyAllSuite(self.config)

        file_model = "/path/to/model_1/model_folder_1/model_{}_{}.joblib"
        file_classify = (
            "/path/to/classify_1/nrt_bo_001/classify_folder_1/classify_report_{}.tsv"
        )
        file_contingency = "/path/to/classify_1/nrt_bo_001/classify_folder_1/classify_contingency_tables_{}.parquet"
        file_metric_plots = "/path/to/classify_1/nrt_bo_001/classify_folder_1/classify_metric_plots_{}.svg"

        # Check model file names (should use composite keys)
        assert file_model.format("xgb", "temp") == str(ds.model_file_names["xgb_temp"])
        assert file_model.format("dt", "psal") == str(ds.model_file_names["dt_psal"])

        # Check aggregated result file names (should use target keys)
        assert file_classify.format("temp") == str(
            ds.output_file_names["report"]["temp"]
        )
        assert file_contingency.format("psal") == str(
            ds.output_file_names["contingency_table"]["psal"]
        )
        assert file_metric_plots.format("pres") == str(
            ds.output_file_names["metric_plot"]["pres"]
        )

    def test_base_model(self):
        """Ensure that the configured base model is a ModelSuite instance."""
        ds = ClassifyAllSuite(self.config)
        assert isinstance(ds.base_model, ModelSuite)


def _setup_datasets(test_obj):
    """Iterate over config files and run prior classification steps to prepare test datasets."""
    test_obj.configs = []
    test_obj.extracts = []
    for x in test_obj.config_file_paths:
        config = ClassificationConfig(x)
        config.select("NRT_BO_001")

        # Inject ModelSuite settings
        config.data["step_class_set"]["steps"]["classify"] = "ClassifyAllSuite"
        config.data["step_class_set"]["steps"]["model"] = "ModelSuite"
        config.data["step_param_set"]["steps"]["model"] = {"methods": ["XGB", "DT"]}

        ds_input = load_classify_step1_input_dataset(config)
        ds_input.input_file_name = str(test_obj.test_data_file)
        ds_input.read_input_data()

        ds_summary = load_classify_step2_summary_dataset(
            config, input_data=ds_input.input_data
        )
        ds_summary.calculate_stats()

        ds_select = load_classify_step3_select_dataset(
            config, input_data=ds_input.input_data
        )
        ds_select.label_profiles()

        ds_locate = load_classify_step4_locate_dataset(
            config,
            input_data=ds_input.input_data,
            selected_profiles=ds_select.selected_profiles,
        )
        ds_locate.process_targets()

        ds_extract = load_classify_step5_extract_dataset(
            config,
            input_data=ds_input.input_data,
            selected_profiles=ds_select.selected_profiles,
            selected_rows=ds_locate.selected_rows,
            summary_stats=ds_summary.summary_stats,
        )
        ds_extract.process_targets()

        test_obj.configs.append(config)
        test_obj.extracts.append(ds_extract)


def _setup_classify_all_suite(test_obj):
    """Set up test environment and file paths for the suite."""
    config_path = Path(__file__).resolve().parent / "data" / "config"
    test_obj.config_file_paths = [
        config_path / "test_classify_001.yaml",
        config_path / "test_classify_002.yaml",
        config_path / "test_classify_003.yaml",
    ]
    test_obj.test_data_file = str(
        Path(__file__).resolve().parent / "data" / "input" / "nrt_cora_bo_test.parquet"
    )
    _setup_datasets(test_obj)

    data_path = Path(__file__).resolve().parent / "data" / "classify"
    test_obj.report_file_names = {
        "temp": str(data_path / "temp_classify_report_temp.tsv"),
        "psal": str(data_path / "temp_classify_report_psal.tsv"),
        "pres": str(data_path / "temp_classify_report_pres.tsv"),
    }
    test_obj.prediction_file_names = {
        "temp": str(data_path / "temp_classify_prediction_temp.parquet"),
        "psal": str(data_path / "temp_classify_prediction_psal.parquet"),
        "pres": str(data_path / "temp_classify_prediction_pres.parquet"),
    }
    test_obj.contingency_table_file_names = {
        "temp": str(data_path / "temp_classify_contingency_tables_temp.parquet"),
        "psal": str(data_path / "temp_classify_contingency_tables_psal.parquet"),
        "pres": str(data_path / "temp_classify_contingency_tables_pres.parquet"),
    }
    test_obj.shap_value_file_names = {
        "temp": str(data_path / "temp_classify_shap_values_temp.parquet"),
        "psal": str(data_path / "temp_classify_shap_values_psal.parquet"),
        "pres": str(data_path / "temp_classify_shap_values_pres.parquet"),
    }
    test_obj.metric_plots_file_names = {
        "temp": str(data_path / "temp_classify_metric_plots_temp.svg"),
        "psal": str(data_path / "temp_classify_metric_plots_psal.svg"),
        "pres": str(data_path / "temp_classify_metric_plots_pres.svg"),
    }


class TestClassifyAllSuite:
    """
    A suite of tests ensuring that the `ClassifyAllSuite` step correctly loads
    multiple models, tests them against input data, and saves aggregated results.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test environment and file maps."""
        _setup_classify_all_suite(self)
        model_path = Path(__file__).resolve().parent / "data" / "training"

        # Map composite keys to existing test models to allow `read_models` to succeed.
        # (It will load the XGB model into the DT wrapper, which is fine for pipeline testing).
        self.model_file_names = {
            "xgb_temp": str(model_path / "model_temp_xgb.joblib"),
            "dt_temp": str(model_path / "model_temp_dt.joblib"),
            "xgb_psal": str(model_path / "model_psal_xgb.joblib"),
            "dt_psal": str(model_path / "model_psal_dt.joblib"),
            "xgb_pres": str(model_path / "model_pres_xgb.joblib"),
            "dt_pres": str(model_path / "model_pres_dt.joblib"),
        }

    @pytest.mark.parametrize("idx", range(TEST_COUNT))
    def test_test_sets(self, idx):
        """Check that test sets are loaded into ClassifyAllSuite correctly."""
        ds = ClassifyAllSuite(
            self.configs[idx],
            test_sets=self.extracts[idx].target_features,
        )

        assert isinstance(ds.test_sets["temp"], pl.DataFrame)
        assert ds.test_sets["temp"].shape[0] == 19480
        assert ds.test_sets["temp"].shape[1] == 56

    @pytest.mark.parametrize("idx", range(TEST_COUNT))
    def test_read_models(self, idx):
        """Confirm that reading models populates the 'models' dict with composite keys."""
        ds = ClassifyAllSuite(
            self.configs[idx],
            test_sets=self.extracts[idx].target_features,
        )
        ds.model_file_names = self.model_file_names
        ds.read_models()

        assert "xgb_temp" in ds.models
        assert "dt_temp" in ds.models

    @pytest.mark.parametrize("idx", range(TEST_COUNT))
    def test_shap_flag(self, idx):
        ds = ClassifyAllSuite(self.configs[idx])
        model = ds.base_model
        assert not model.enable_shap
        for method_obj in model.method_objs.values():
            assert not method_obj.enable_shap

        self.configs[idx].data["step_param_set"]["steps"]["model"]["calculate_shap"] = (
            True
        )
        ds = ClassifyAllSuite(self.configs[idx])
        model = ds.base_model
        assert model.enable_shap
        for method_obj in model.method_objs.values():
            assert method_obj.enable_shap

        self.configs[idx].data["step_param_set"]["steps"]["model"]["calculate_shap"] = (
            False
        )
        ds = ClassifyAllSuite(self.configs[idx])
        model = ds.base_model
        assert not model.enable_shap
        for method_obj in model.method_objs.values():
            assert not method_obj.enable_shap

    @pytest.mark.parametrize("idx", range(TEST_COUNT))
    def test_with_models(self, idx):
        """
        Check that testing targets populates aggregated result columns and
        contingency tables. Dimensions should reflect multiple models.
        """
        ds = ClassifyAllSuite(
            self.configs[idx],
            test_sets=self.extracts[idx].target_features,
        )
        ds.model_file_names = self.model_file_names
        ds.read_models()
        ds.test_targets()

        # Predictions: 19480 rows * 2 models = 38960. Should include 'method' column.
        assert isinstance(ds.predictions["temp"], pl.DataFrame)
        assert ds.predictions["temp"].shape[0] == 38960
        assert "method" in ds.predictions["temp"].columns

        # Contingency Tables
        assert isinstance(ds.contingency_tables["psal"], pl.DataFrame)
        assert ds.contingency_tables["psal"].height == 38960
        assert "method" in ds.contingency_tables["psal"].columns

    @pytest.mark.parametrize("idx", range(TEST_COUNT))
    def test_without_model(self, idx):
        """Ensure that testing without loaded models raises a ValueError."""
        ds = ClassifyAllSuite(
            self.configs[idx],
            test_sets=self.extracts[idx].target_features,
        )
        with pytest.raises(ValueError):
            ds.test_targets()

    @pytest.mark.parametrize("idx", range(TEST_COUNT))
    def test_write_reports(self, idx):
        """Verify that aggregated test reports are correctly written to file."""
        ds = ClassifyAllSuite(
            self.configs[idx],
            test_sets=self.extracts[idx].target_features,
        )
        ds.model_file_names = self.model_file_names
        ds.output_file_names["report"] = self.report_file_names
        ds.read_models()
        ds.test_targets()
        ds.write_reports()

        assert os.path.exists(ds.output_file_names["report"]["temp"])
        assert os.path.exists(ds.output_file_names["report"]["psal"])

        os.remove(ds.output_file_names["report"]["temp"])
        os.remove(ds.output_file_names["report"]["psal"])
        os.remove(ds.output_file_names["report"]["pres"])

    @pytest.mark.parametrize("idx", range(TEST_COUNT))
    def test_write_contingency_tables(self, idx):
        """Verify that aggregated contingency tables are correctly written to file."""
        ds = ClassifyAllSuite(
            self.configs[idx],
            test_sets=self.extracts[idx].target_features,
        )
        ds.model_file_names = self.model_file_names
        ds.output_file_names["contingency_table"] = self.contingency_table_file_names

        ds.read_models()
        ds.test_targets()
        ds.write_contingency_tables()

        assert os.path.exists(ds.output_file_names["contingency_table"]["temp"])

        os.remove(ds.output_file_names["contingency_table"]["temp"])
        os.remove(ds.output_file_names["contingency_table"]["psal"])
        os.remove(ds.output_file_names["contingency_table"]["pres"])

    @pytest.mark.parametrize("idx", range(TEST_COUNT))
    def test_write_shap_values(self, idx):
        """Verify that aggregated contingency tables are correctly written to file."""
        self.configs[idx].data["step_param_set"]["steps"]["model"]["calculate_shap"] = (
            True
        )
        ds = ClassifyAllSuite(
            self.configs[idx],
            test_sets=self.extracts[idx].target_features,
        )
        ds.model_file_names = self.model_file_names
        ds.output_file_names["shap_value"] = self.shap_value_file_names

        ds.read_models()
        ds.test_targets()
        ds.write_shap_values()

        assert os.path.exists(ds.output_file_names["shap_value"]["temp"])

        os.remove(ds.output_file_names["shap_value"]["temp"])
        os.remove(ds.output_file_names["shap_value"]["psal"])
        os.remove(ds.output_file_names["shap_value"]["pres"])

    @pytest.mark.parametrize("idx", range(TEST_COUNT))
    def test_create_metric_plot(self, idx):
        """Verify that multi-method roc and prc plots are correctly written to file."""
        ds = ClassifyAllSuite(
            self.configs[idx],
            test_sets=self.extracts[idx].target_features,
        )
        ds.model_file_names = self.model_file_names
        ds.output_file_names["metric_plot"] = self.metric_plots_file_names

        ds.read_models()
        ds.test_targets()
        ds.create_metric_plots()

        assert os.path.exists(ds.output_file_names["metric_plot"]["temp"])

        os.remove(ds.output_file_names["metric_plot"]["temp"])
        os.remove(ds.output_file_names["metric_plot"]["psal"])
        os.remove(ds.output_file_names["metric_plot"]["pres"])

    @pytest.mark.parametrize("idx", range(TEST_COUNT))
    def test_write_no_results(self, idx):
        """Ensure ValueError is raised if write_reports is called without test results."""
        ds = ClassifyAllSuite(
            self.configs[idx],
            test_sets=self.extracts[idx].target_features,
        )
        with pytest.raises(ValueError):
            ds.write_reports()

    @pytest.mark.parametrize("idx", range(TEST_COUNT))
    def test_read_models_no_file(self, idx):
        """Check that FileNotFoundError is raised if model files are missing during loading."""
        ds = ClassifyAllSuite(
            self.configs[idx],
            test_sets=self.extracts[idx].target_features,
        )
        data_path = Path(__file__).resolve().parent / "data" / "training"
        ds.model_file_names["xgb_temp"] = str(data_path / "model_does_not_exist.joblib")

        with pytest.raises(FileNotFoundError):
            ds.read_models()

    @pytest.mark.parametrize("idx", range(TEST_COUNT))
    def test_write_predictions(self, idx):
        """Verify that aggregated test predictions are correctly written to file."""
        ds = ClassifyAllSuite(
            self.configs[idx],
            test_sets=self.extracts[idx].target_features,
        )
        ds.model_file_names = self.model_file_names
        ds.output_file_names["prediction"] = self.prediction_file_names
        ds.read_models()
        ds.test_targets()
        ds.write_predictions()

        assert os.path.exists(ds.output_file_names["prediction"]["temp"])

        os.remove(ds.output_file_names["prediction"]["temp"])
        os.remove(ds.output_file_names["prediction"]["psal"])
        os.remove(ds.output_file_names["prediction"]["pres"])
