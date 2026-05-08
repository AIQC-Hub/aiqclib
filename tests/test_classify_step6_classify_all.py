"""
This module contains unit tests for the `ClassifyAll` class,
focusing on its functionality for loading, testing, and saving
XGBoost models and their results within the classification workflow.
"""

import os
import pytest
from pathlib import Path

import polars as pl

from aiqclib.classify.step6_classify_dataset.dataset_all import ClassifyAll
from aiqclib.common.config.classify_config import ClassificationConfig
from aiqclib.common.loader.classify_loader import (
    load_classify_step1_input_dataset,
    load_classify_step2_summary_dataset,
    load_classify_step3_select_dataset,
    load_classify_step4_locate_dataset,
    load_classify_step5_extract_dataset,
)
from aiqclib.train.models.logistic_regression import LogisticRegression
from aiqclib.train.models.linear_discriminant_analysis import LinearDiscriminantAnalysis
from aiqclib.train.models.support_vector_machine import SupportVectorMachine
from aiqclib.train.models.decision_tree import DecisionTree
from aiqclib.train.models.random_forest import RandomForest
from aiqclib.train.models.xgboost import XGBoost
from aiqclib.train.models.k_nearest_neighbors import KNearestNeighbors
from aiqclib.train.models.gaussian_naive_bayes import GaussianNaiveBayes
from aiqclib.train.models.multilayer_perceptron import MultilayerPerceptron

TEST_COUNT = 3
METHOD_TEST_COUNT = 1


class TestClassifyAllClass:
    """
    A suite of tests ensuring that the `ClassifyAll` step correctly loads models,
    tests them against input data, and saves classification reports and predictions.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test environment and load input, summary, select, and locate data."""
        config_path = Path(__file__).resolve().parent / "data" / "config"
        self.config_file = config_path / "test_classify_001.yaml"
        self.config = ClassificationConfig(self.config_file)
        self.config.select("NRT_BO_001")

        self.config_file_3 = config_path / "test_classify_003.yaml"
        self.config_3 = ClassificationConfig(self.config_file_3)
        self.config_3.select("NRT_BO_001")

    def test_step_name(self):
        """Check that the ClassifyAll step name is correctly assigned."""
        ds = ClassifyAll(self.config)
        assert ds.step_name == "classify"

    def test_output_file_names(self):
        """Verify that default output file names (model, results, contingency tables) are as expected."""
        ds = ClassifyAll(self.config)

        file_model = "/path/to/model_1/model_folder_1/model_{}.joblib"
        file_classify = (
            "/path/to/classify_1/nrt_bo_001/classify_folder_1/classify_report_{}.tsv"
        )
        file_contingency = "/path/to/classify_1/nrt_bo_001/classify_folder_1/classify_contingency_tables_{}.parquet"
        file_metric_plots = "/path/to/classify_1/nrt_bo_001/classify_folder_1/classify_metric_plots_{}.svg"

        # Check model file names
        assert file_model.format("temp") == str(ds.model_file_names["temp"])
        assert file_model.format("psal") == str(ds.model_file_names["psal"])
        assert file_model.format("pres") == str(ds.model_file_names["pres"])

        # Check report file names
        assert file_classify.format("temp") == str(
            ds.output_file_names["report"]["temp"]
        )
        assert file_classify.format("psal") == str(
            ds.output_file_names["report"]["psal"]
        )
        assert file_classify.format("pres") == str(
            ds.output_file_names["report"]["pres"]
        )

        # Check contingency table file names
        assert file_contingency.format("temp") == str(
            ds.output_file_names["contingency_table"]["temp"]
        )
        assert file_contingency.format("psal") == str(
            ds.output_file_names["contingency_table"]["psal"]
        )
        assert file_contingency.format("pres") == str(
            ds.output_file_names["contingency_table"]["pres"]
        )

        # # Check metric plot file names
        assert file_metric_plots.format("temp") == str(
            ds.output_file_names["metric_plot"]["temp"]
        )
        assert file_metric_plots.format("psal") == str(
            ds.output_file_names["metric_plot"]["psal"]
        )
        assert file_metric_plots.format("pres") == str(
            ds.output_file_names["metric_plot"]["pres"]
        )

    def test_base_model(self):
        """Ensure that the configured base model is an XGBoost instance."""
        ds = ClassifyAll(self.config)
        assert isinstance(ds.base_model, XGBoost)

    def test_shap_flag(self):
        ds = ClassifyAll(self.config)
        model = ds.base_model
        assert not model.enable_shap

        self.config.data["step_param_set"]["steps"]["model"]["calculate_shap"] = True
        ds = ClassifyAll(self.config)
        model = ds.base_model
        assert model.enable_shap

        self.config.data["step_param_set"]["steps"]["model"]["calculate_shap"] = False
        ds = ClassifyAll(self.config)
        model = ds.base_model
        assert not model.enable_shap

    def test_logistic_regression_model(self):
        """Ensure that the configured base model is a LogisticRegression instance."""
        self.config.data["step_class_set"]["steps"]["model"] = "LogisticRegression"
        ds = ClassifyAll(self.config)
        assert isinstance(ds.base_model, LogisticRegression)

    def test_nthreads(self):
        ds = ClassifyAll(self.config)
        assert ds.base_model.model_params["n_jobs"] == -1

        ds_3 = ClassifyAll(self.config_3)
        assert ds_3.base_model.model_params["n_jobs"] == 2


def _setup_datasets(test_obj):
    test_obj.configs = []
    test_obj.extracts = []
    for x in test_obj.config_file_paths:
        config = ClassificationConfig(x)
        config.select("NRT_BO_001")

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


def _setup_classify_all(test_obj):
    """Set up test environment and load input, summary, select, and locate data."""
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
    test_obj.n_jobs = [-1, -1, 2]


class TestClassifyAll:
    """
    A suite of tests ensuring that the `ClassifyAll` step correctly loads models,
    tests them against input data, and saves classification reports and predictions.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test environment and load input, summary, select, and locate data."""
        _setup_classify_all(self)
        model_path = Path(__file__).resolve().parent / "data" / "training"
        self.model_file_names = {
            "temp": str(model_path / "model_temp.joblib"),
            "psal": str(model_path / "model_psal.joblib"),
            "pres": str(model_path / "model_pres.joblib"),
        }

    @pytest.mark.parametrize("idx", range(TEST_COUNT))
    def test_test_sets(self, idx):
        """Check that test sets are loaded into ClassifyAll correctly."""
        ds = ClassifyAll(
            self.configs[idx],
            test_sets=self.extracts[idx].target_features,
        )

        assert isinstance(ds.test_sets["temp"], pl.DataFrame)
        assert ds.test_sets["temp"].shape[0] == 19480
        assert ds.test_sets["temp"].shape[1] == 56

        assert isinstance(ds.test_sets["psal"], pl.DataFrame)
        assert ds.test_sets["psal"].shape[0] == 19480
        assert ds.test_sets["psal"].shape[1] == 56

        assert isinstance(ds.test_sets["pres"], pl.DataFrame)
        assert ds.test_sets["pres"].shape[0] == 19480
        assert ds.test_sets["pres"].shape[1] == 56

    @pytest.mark.parametrize("idx", range(TEST_COUNT))
    def test_read_models(self, idx):
        """Confirm that reading models populates the 'models' dictionary with XGBoost instances."""
        ds = ClassifyAll(
            self.configs[idx],
            test_sets=self.extracts[idx].target_features,
        )
        ds.model_file_names = self.model_file_names
        ds.read_models()

        assert isinstance(ds.models["temp"], XGBoost)
        assert isinstance(ds.models["psal"], XGBoost)
        assert isinstance(ds.models["pres"], XGBoost)

    @pytest.mark.parametrize("idx", range(TEST_COUNT))
    def test_with_xgboost(self, idx):
        """Check that testing targets after model loading populates the result columns and contingency tables."""
        ds = ClassifyAll(
            self.configs[idx],
            test_sets=self.extracts[idx].target_features,
        )
        ds.model_file_names = self.model_file_names
        ds.read_models()
        ds.test_targets()

        # Check Test Sets
        assert isinstance(ds.test_sets["temp"], pl.DataFrame)
        assert ds.test_sets["temp"].shape[0] == 19480
        assert ds.test_sets["temp"].shape[1] == 56

        assert isinstance(ds.test_sets["psal"], pl.DataFrame)
        assert ds.test_sets["psal"].shape[0] == 19480
        assert ds.test_sets["psal"].shape[1] == 56

        assert isinstance(ds.test_sets["pres"], pl.DataFrame)
        assert ds.test_sets["pres"].shape[0] == 19480
        assert ds.test_sets["pres"].shape[1] == 56

        # Check Contingency Tables
        assert isinstance(ds.contingency_tables["temp"], pl.DataFrame)
        assert ds.contingency_tables["temp"].height == 19480
        assert ds.contingency_tables["temp"].columns == [
            "k",
            "label",
            "predicted_label",
            "score",
        ]

        assert isinstance(ds.contingency_tables["psal"], pl.DataFrame)
        assert ds.contingency_tables["psal"].height == 19480

        assert isinstance(ds.contingency_tables["pres"], pl.DataFrame)
        assert ds.contingency_tables["pres"].height == 19480

    @pytest.mark.parametrize("idx", range(TEST_COUNT))
    def test_without_model(self, idx):
        """Ensure that testing without loaded models raises a ValueError."""
        ds = ClassifyAll(
            self.configs[idx],
            test_sets=self.extracts[idx].target_features,
        )
        with pytest.raises(ValueError):
            ds.test_targets()

    @pytest.mark.parametrize("idx", range(TEST_COUNT))
    def test_write_reports(self, idx):
        """Verify that test reports are correctly written to file."""
        ds = ClassifyAll(
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
        assert os.path.exists(ds.output_file_names["report"]["pres"])

        os.remove(ds.output_file_names["report"]["temp"])
        os.remove(ds.output_file_names["report"]["psal"])
        os.remove(ds.output_file_names["report"]["pres"])

    @pytest.mark.parametrize("idx", range(TEST_COUNT))
    def test_write_contingency_tables(self, idx):
        """Verify that contingency tables are correctly written to file."""
        ds = ClassifyAll(
            self.configs[idx],
            test_sets=self.extracts[idx].target_features,
        )
        ds.model_file_names = self.model_file_names
        # Override output file path for testing
        ds.output_file_names["contingency_table"] = self.contingency_table_file_names

        ds.read_models()
        ds.test_targets()
        ds.write_contingency_tables()

        assert os.path.exists(ds.output_file_names["contingency_table"]["temp"])
        assert os.path.exists(ds.output_file_names["contingency_table"]["psal"])
        assert os.path.exists(ds.output_file_names["contingency_table"]["pres"])

        os.remove(ds.output_file_names["contingency_table"]["temp"])
        os.remove(ds.output_file_names["contingency_table"]["psal"])
        os.remove(ds.output_file_names["contingency_table"]["pres"])

    @pytest.mark.parametrize("idx", range(TEST_COUNT))
    def test_write_shap_values(self, idx):
        """
        Check that contingency tables are correctly written to file,
        and then remove the temporary files created.
        """
        self.configs[idx].data["step_param_set"]["steps"]["model"]["calculate_shap"] = (
            True
        )
        ds = ClassifyAll(
            self.configs[idx],
            test_sets=self.extracts[idx].target_features,
        )
        ds.model_file_names = self.model_file_names
        # Override output file path for testing
        ds.output_file_names["shap_value"] = self.shap_value_file_names

        ds.read_models()
        ds.test_targets()
        ds.write_shap_values()

        assert os.path.exists(ds.output_file_names["shap_value"]["temp"])
        assert os.path.exists(ds.output_file_names["shap_value"]["psal"])
        assert os.path.exists(ds.output_file_names["shap_value"]["pres"])

        os.remove(ds.output_file_names["shap_value"]["temp"])
        os.remove(ds.output_file_names["shap_value"]["psal"])
        os.remove(ds.output_file_names["shap_value"]["pres"])

    @pytest.mark.parametrize("idx", range(TEST_COUNT))
    def test_create_metric_plot(self, idx):
        """Verify that roc and prc plots are correctly written to file."""
        ds = ClassifyAll(
            self.configs[idx],
            test_sets=self.extracts[idx].target_features,
        )
        ds.model_file_names = self.model_file_names
        # Override output file path for testing
        ds.output_file_names["metric_plot"] = self.metric_plots_file_names

        ds.read_models()
        ds.test_targets()
        ds.create_metric_plots()

        assert os.path.exists(ds.output_file_names["metric_plot"]["temp"])
        assert os.path.exists(ds.output_file_names["metric_plot"]["psal"])
        assert os.path.exists(ds.output_file_names["metric_plot"]["pres"])

        os.remove(ds.output_file_names["metric_plot"]["temp"])
        os.remove(ds.output_file_names["metric_plot"]["psal"])
        os.remove(ds.output_file_names["metric_plot"]["pres"])

    @pytest.mark.parametrize("idx", range(TEST_COUNT))
    def test_write_no_results(self, idx):
        """Ensure ValueError is raised if write_reports is called without test results."""
        ds = ClassifyAll(
            self.configs[idx],
            test_sets=self.extracts[idx].target_features,
        )
        with pytest.raises(ValueError):
            ds.write_reports()

    @pytest.mark.parametrize("idx", range(TEST_COUNT))
    def test_write_no_contingency_tables(self, idx):
        """Ensure ValueError is raised if write_contingency_tables is called without results."""
        ds = ClassifyAll(
            self.configs[idx],
            test_sets=self.extracts[idx].target_features,
        )
        with pytest.raises(ValueError):
            ds.write_contingency_tables()

    @pytest.mark.parametrize("idx", range(TEST_COUNT))
    def test_create_no_metric_plots(self, idx):
        """Ensure ValueError is raised if create_metric_plots is called without results."""
        ds = ClassifyAll(
            self.configs[idx],
            test_sets=self.extracts[idx].target_features,
        )
        with pytest.raises(ValueError):
            ds.create_metric_plots()

    @pytest.mark.parametrize("idx", range(TEST_COUNT))
    def test_read_models_no_file(self, idx):
        """Check that FileNotFoundError is raised if model files are missing during loading."""
        ds = ClassifyAll(
            self.configs[idx],
            test_sets=self.extracts[idx].target_features,
        )
        data_path = Path(__file__).resolve().parent / "data" / "training"
        ds.model_file_names["temp"] = str(data_path / "model.joblib")
        ds.model_file_names["psal"] = str(data_path / "model.joblib")
        ds.model_file_names["pres"] = str(data_path / "model.joblib")

        with pytest.raises(FileNotFoundError):
            ds.read_models()

    @pytest.mark.parametrize("idx", range(TEST_COUNT))
    def test_n_jobs(self, idx):
        """Confirm that reading models populates the 'models' dictionary with XGBoost instances."""
        ds = ClassifyAll(
            self.configs[idx],
            test_sets=self.extracts[idx].target_features,
        )
        ds.model_file_names = self.model_file_names
        ds.read_models()

        assert ds.models["temp"].model_params["n_jobs"] == self.n_jobs[idx]
        assert ds.models["psal"].model_params["n_jobs"] == self.n_jobs[idx]
        assert ds.models["pres"].model_params["n_jobs"] == self.n_jobs[idx]

        assert ds.models["temp"].model.n_jobs == self.n_jobs[idx]
        assert ds.models["psal"].model.n_jobs == self.n_jobs[idx]
        assert ds.models["pres"].model.n_jobs == self.n_jobs[idx]

    @pytest.mark.parametrize("idx", range(TEST_COUNT))
    def test_write_predictions(self, idx):
        """Verify that test predictions are correctly written to file."""
        ds = ClassifyAll(
            self.configs[idx],
            test_sets=self.extracts[idx].target_features,
        )
        ds.model_file_names = self.model_file_names
        ds.output_file_names["prediction"] = self.prediction_file_names
        ds.read_models()
        ds.test_targets()
        ds.write_predictions()

        assert os.path.exists(ds.output_file_names["prediction"]["temp"])
        assert os.path.exists(ds.output_file_names["prediction"]["psal"])
        assert os.path.exists(ds.output_file_names["prediction"]["pres"])

        os.remove(ds.output_file_names["prediction"]["temp"])
        os.remove(ds.output_file_names["prediction"]["psal"])
        os.remove(ds.output_file_names["prediction"]["pres"])


class TestXGBoost:
    """Tests for the XGBoost model wrapper."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test environment and load input, summary, select, and locate data."""
        _setup_classify_all(self)
        model_path = Path(__file__).resolve().parent / "data" / "training"
        self.model_file_names = {
            "temp": str(model_path / "model_temp_xgb.joblib"),
            "psal": str(model_path / "model_psal_xgb.joblib"),
            "pres": str(model_path / "model_pres_xgb.joblib"),
        }

        for x in self.configs:
            x.data["step_class_set"]["steps"]["model"] = "XGBoost"

    @pytest.mark.parametrize("idx", range(METHOD_TEST_COUNT))
    def test_read_models(self, idx):
        """Confirm that reading models populates the 'models' dictionary with XGBoost instances."""
        ds = ClassifyAll(
            self.configs[idx],
            test_sets=self.extracts[idx].target_features,
        )
        ds.model_file_names = self.model_file_names
        ds.read_models()

        assert isinstance(ds.models["temp"], XGBoost)
        assert isinstance(ds.models["psal"], XGBoost)
        assert isinstance(ds.models["pres"], XGBoost)

    @pytest.mark.parametrize("idx", range(METHOD_TEST_COUNT))
    def test_with_xgboost(self, idx):
        """Check that testing targets after model loading populates the result columns and contingency tables."""
        ds = ClassifyAll(
            self.configs[idx],
            test_sets=self.extracts[idx].target_features,
        )
        ds.model_file_names = self.model_file_names
        ds.read_models()
        ds.test_targets()

        # Check Test Sets
        assert isinstance(ds.test_sets["temp"], pl.DataFrame)
        assert ds.test_sets["temp"].shape[0] == 19480
        assert ds.test_sets["temp"].shape[1] == 56

        assert isinstance(ds.test_sets["psal"], pl.DataFrame)
        assert ds.test_sets["psal"].shape[0] == 19480
        assert ds.test_sets["psal"].shape[1] == 56

        assert isinstance(ds.test_sets["pres"], pl.DataFrame)
        assert ds.test_sets["pres"].shape[0] == 19480
        assert ds.test_sets["pres"].shape[1] == 56

        # Check Contingency Tables
        assert isinstance(ds.contingency_tables["temp"], pl.DataFrame)
        assert ds.contingency_tables["temp"].height == 19480
        assert ds.contingency_tables["temp"].columns == [
            "k",
            "label",
            "predicted_label",
            "score",
        ]

        assert isinstance(ds.contingency_tables["psal"], pl.DataFrame)
        assert ds.contingency_tables["psal"].height == 19480

        assert isinstance(ds.contingency_tables["pres"], pl.DataFrame)
        assert ds.contingency_tables["pres"].height == 19480


class TestLogisticRegression:
    """Tests for the Logistic Regression model wrapper."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test environment and load input, summary, select, and locate data."""
        _setup_classify_all(self)
        model_path = Path(__file__).resolve().parent / "data" / "training"
        self.model_file_names = {
            "temp": str(model_path / "model_temp_logit.joblib"),
            "psal": str(model_path / "model_psal_logit.joblib"),
            "pres": str(model_path / "model_pres_logit.joblib"),
        }

        for x in self.configs:
            x.data["step_class_set"]["steps"]["model"] = "LogisticRegression"

    @pytest.mark.parametrize("idx", range(METHOD_TEST_COUNT))
    def test_read_models(self, idx):
        """Confirm that reading models populates the 'models' dictionary with LogisticRegression instances."""
        ds = ClassifyAll(
            self.configs[idx],
            test_sets=self.extracts[idx].target_features,
        )
        ds.model_file_names = self.model_file_names
        ds.read_models()

        assert isinstance(ds.models["temp"], LogisticRegression)
        assert isinstance(ds.models["psal"], LogisticRegression)
        assert isinstance(ds.models["pres"], LogisticRegression)

    @pytest.mark.parametrize("idx", range(METHOD_TEST_COUNT))
    def test_with_logit(self, idx):
        """Check that testing targets after model loading populates the result columns and contingency tables."""
        ds = ClassifyAll(
            self.configs[idx],
            test_sets=self.extracts[idx].target_features,
        )
        ds.model_file_names = self.model_file_names
        ds.read_models()
        ds.test_targets()

        # Check Test Sets
        assert isinstance(ds.test_sets["temp"], pl.DataFrame)
        assert ds.test_sets["temp"].shape[0] == 19480
        assert ds.test_sets["temp"].shape[1] == 56

        assert isinstance(ds.test_sets["psal"], pl.DataFrame)
        assert ds.test_sets["psal"].shape[0] == 19480
        assert ds.test_sets["psal"].shape[1] == 56

        assert isinstance(ds.test_sets["pres"], pl.DataFrame)
        assert ds.test_sets["pres"].shape[0] == 19480
        assert ds.test_sets["pres"].shape[1] == 56

        # Check Contingency Tables
        assert isinstance(ds.contingency_tables["temp"], pl.DataFrame)
        assert ds.contingency_tables["temp"].height == 19480
        assert ds.contingency_tables["temp"].columns == [
            "k",
            "label",
            "predicted_label",
            "score",
        ]

        assert isinstance(ds.contingency_tables["psal"], pl.DataFrame)
        assert ds.contingency_tables["psal"].height == 19480

        assert isinstance(ds.contingency_tables["pres"], pl.DataFrame)
        assert ds.contingency_tables["pres"].height == 19480


class TestLDA:
    """Tests for the Linear Discriminant Analysis model wrapper."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test environment and load input, summary, select, and locate data."""
        _setup_classify_all(self)
        model_path = Path(__file__).resolve().parent / "data" / "training"
        self.model_file_names = {
            "temp": str(model_path / "model_temp_lda.joblib"),
            "psal": str(model_path / "model_psal_lda.joblib"),
            "pres": str(model_path / "model_pres_lda.joblib"),
        }

        for x in self.configs:
            x.data["step_class_set"]["steps"]["model"] = "LinearDiscriminantAnalysis"

    @pytest.mark.parametrize("idx", range(METHOD_TEST_COUNT))
    def test_read_models(self, idx):
        """Confirm that reading models populates the 'models' dictionary with LinearDiscriminantAnalysis instances."""
        ds = ClassifyAll(
            self.configs[idx],
            test_sets=self.extracts[idx].target_features,
        )
        ds.model_file_names = self.model_file_names
        ds.read_models()

        assert isinstance(ds.models["temp"], LinearDiscriminantAnalysis)
        assert isinstance(ds.models["psal"], LinearDiscriminantAnalysis)
        assert isinstance(ds.models["pres"], LinearDiscriminantAnalysis)

    @pytest.mark.parametrize("idx", range(METHOD_TEST_COUNT))
    def test_with_lda(self, idx):
        """Check that testing targets after model loading populates the result columns and contingency tables."""
        ds = ClassifyAll(
            self.configs[idx],
            test_sets=self.extracts[idx].target_features,
        )
        ds.model_file_names = self.model_file_names
        ds.read_models()
        ds.test_targets()

        # Check Test Sets
        assert isinstance(ds.test_sets["temp"], pl.DataFrame)
        assert ds.test_sets["temp"].shape[0] == 19480
        assert ds.test_sets["temp"].shape[1] == 56

        assert isinstance(ds.test_sets["psal"], pl.DataFrame)
        assert ds.test_sets["psal"].shape[0] == 19480
        assert ds.test_sets["psal"].shape[1] == 56

        assert isinstance(ds.test_sets["pres"], pl.DataFrame)
        assert ds.test_sets["pres"].shape[0] == 19480
        assert ds.test_sets["pres"].shape[1] == 56

        # Check Contingency Tables
        assert isinstance(ds.contingency_tables["temp"], pl.DataFrame)
        assert ds.contingency_tables["temp"].height == 19480
        assert ds.contingency_tables["temp"].columns == [
            "k",
            "label",
            "predicted_label",
            "score",
        ]

        assert isinstance(ds.contingency_tables["psal"], pl.DataFrame)
        assert ds.contingency_tables["psal"].height == 19480

        assert isinstance(ds.contingency_tables["pres"], pl.DataFrame)
        assert ds.contingency_tables["pres"].height == 19480


class TestSVM:
    """Tests for the SVM model wrapper."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test environment and load input, summary, select, and locate data."""
        _setup_classify_all(self)
        model_path = Path(__file__).resolve().parent / "data" / "training"
        self.model_file_names = {
            "temp": str(model_path / "model_temp_svm.joblib"),
            "psal": str(model_path / "model_psal_svm.joblib"),
            "pres": str(model_path / "model_pres_svm.joblib"),
        }

        for x in self.configs:
            x.data["step_class_set"]["steps"]["model"] = "SVM"

    @pytest.mark.parametrize("idx", range(METHOD_TEST_COUNT))
    def test_read_models(self, idx):
        """Confirm that reading models populates the 'models' dictionary with SVM instances."""
        ds = ClassifyAll(
            self.configs[idx],
            test_sets=self.extracts[idx].target_features,
        )
        ds.model_file_names = self.model_file_names
        ds.read_models()

        assert isinstance(ds.models["temp"], SupportVectorMachine)
        assert isinstance(ds.models["psal"], SupportVectorMachine)
        assert isinstance(ds.models["pres"], SupportVectorMachine)

    @pytest.mark.parametrize("idx", range(METHOD_TEST_COUNT))
    def test_with_svm(self, idx):
        """Check that testing targets after model loading populates the result columns and contingency tables."""
        ds = ClassifyAll(
            self.configs[idx],
            test_sets=self.extracts[idx].target_features,
        )
        ds.model_file_names = self.model_file_names
        ds.read_models()
        ds.test_targets()

        # Check Test Sets
        assert isinstance(ds.test_sets["temp"], pl.DataFrame)
        assert ds.test_sets["temp"].shape[0] == 19480
        assert ds.test_sets["temp"].shape[1] == 56

        assert isinstance(ds.test_sets["psal"], pl.DataFrame)
        assert ds.test_sets["psal"].shape[0] == 19480
        assert ds.test_sets["psal"].shape[1] == 56

        assert isinstance(ds.test_sets["pres"], pl.DataFrame)
        assert ds.test_sets["pres"].shape[0] == 19480
        assert ds.test_sets["pres"].shape[1] == 56

        # Check Contingency Tables
        assert isinstance(ds.contingency_tables["temp"], pl.DataFrame)
        assert ds.contingency_tables["temp"].height == 19480
        assert ds.contingency_tables["temp"].columns == [
            "k",
            "label",
            "predicted_label",
            "score",
        ]

        assert isinstance(ds.contingency_tables["psal"], pl.DataFrame)
        assert ds.contingency_tables["psal"].height == 19480

        assert isinstance(ds.contingency_tables["pres"], pl.DataFrame)
        assert ds.contingency_tables["pres"].height == 19480


class TestDecisionTree:
    """Tests for the Decision Tree model wrapper."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test environment and load input, summary, select, and locate data."""
        _setup_classify_all(self)
        model_path = Path(__file__).resolve().parent / "data" / "training"
        self.model_file_names = {
            "temp": str(model_path / "model_temp_dt.joblib"),
            "psal": str(model_path / "model_psal_dt.joblib"),
            "pres": str(model_path / "model_pres_dt.joblib"),
        }

        for x in self.configs:
            x.data["step_class_set"]["steps"]["model"] = "DecisionTree"

    @pytest.mark.parametrize("idx", range(METHOD_TEST_COUNT))
    def test_read_models(self, idx):
        """Confirm that reading models populates the 'models' dictionary with DecisionTree instances."""
        ds = ClassifyAll(
            self.configs[idx],
            test_sets=self.extracts[idx].target_features,
        )
        ds.model_file_names = self.model_file_names
        ds.read_models()

        assert isinstance(ds.models["temp"], DecisionTree)
        assert isinstance(ds.models["psal"], DecisionTree)
        assert isinstance(ds.models["pres"], DecisionTree)

    @pytest.mark.parametrize("idx", range(METHOD_TEST_COUNT))
    def test_with_dt(self, idx):
        """Check that testing targets after model loading populates the result columns and contingency tables."""
        ds = ClassifyAll(
            self.configs[idx],
            test_sets=self.extracts[idx].target_features,
        )
        ds.model_file_names = self.model_file_names
        ds.read_models()
        ds.test_targets()

        # Check Test Sets
        assert isinstance(ds.test_sets["temp"], pl.DataFrame)
        assert ds.test_sets["temp"].shape[0] == 19480
        assert ds.test_sets["temp"].shape[1] == 56

        assert isinstance(ds.test_sets["psal"], pl.DataFrame)
        assert ds.test_sets["psal"].shape[0] == 19480
        assert ds.test_sets["psal"].shape[1] == 56

        assert isinstance(ds.test_sets["pres"], pl.DataFrame)
        assert ds.test_sets["pres"].shape[0] == 19480
        assert ds.test_sets["pres"].shape[1] == 56

        # Check Contingency Tables
        assert isinstance(ds.contingency_tables["temp"], pl.DataFrame)
        assert ds.contingency_tables["temp"].height == 19480
        assert ds.contingency_tables["temp"].columns == [
            "k",
            "label",
            "predicted_label",
            "score",
        ]

        assert isinstance(ds.contingency_tables["psal"], pl.DataFrame)
        assert ds.contingency_tables["psal"].height == 19480

        assert isinstance(ds.contingency_tables["pres"], pl.DataFrame)
        assert ds.contingency_tables["pres"].height == 19480


class TestRandomForest:
    """Tests for the Random Forest model wrapper."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test environment and load input, summary, select, and locate data."""
        _setup_classify_all(self)
        model_path = Path(__file__).resolve().parent / "data" / "training"
        self.model_file_names = {
            "temp": str(model_path / "model_temp_rf.joblib"),
            "psal": str(model_path / "model_psal_rf.joblib"),
            "pres": str(model_path / "model_pres_rf.joblib"),
        }

        for x in self.configs:
            x.data["step_class_set"]["steps"]["model"] = "RandomForest"

    @pytest.mark.parametrize("idx", range(METHOD_TEST_COUNT))
    def test_read_models(self, idx):
        """Confirm that reading models populates the 'models' dictionary with RandomForest instances."""
        ds = ClassifyAll(
            self.configs[idx],
            test_sets=self.extracts[idx].target_features,
        )
        ds.model_file_names = self.model_file_names
        ds.read_models()

        assert isinstance(ds.models["temp"], RandomForest)
        assert isinstance(ds.models["psal"], RandomForest)
        assert isinstance(ds.models["pres"], RandomForest)

    @pytest.mark.parametrize("idx", range(METHOD_TEST_COUNT))
    def test_with_rf(self, idx):
        """Check that testing targets after model loading populates the result columns and contingency tables."""
        ds = ClassifyAll(
            self.configs[idx],
            test_sets=self.extracts[idx].target_features,
        )
        ds.model_file_names = self.model_file_names
        ds.read_models()
        ds.test_targets()

        # Check Test Sets
        assert isinstance(ds.test_sets["temp"], pl.DataFrame)
        assert ds.test_sets["temp"].shape[0] == 19480
        assert ds.test_sets["temp"].shape[1] == 56

        assert isinstance(ds.test_sets["psal"], pl.DataFrame)
        assert ds.test_sets["psal"].shape[0] == 19480
        assert ds.test_sets["psal"].shape[1] == 56

        assert isinstance(ds.test_sets["pres"], pl.DataFrame)
        assert ds.test_sets["pres"].shape[0] == 19480
        assert ds.test_sets["pres"].shape[1] == 56

        # Check Contingency Tables
        assert isinstance(ds.contingency_tables["temp"], pl.DataFrame)
        assert ds.contingency_tables["temp"].height == 19480
        assert ds.contingency_tables["temp"].columns == [
            "k",
            "label",
            "predicted_label",
            "score",
        ]

        assert isinstance(ds.contingency_tables["psal"], pl.DataFrame)
        assert ds.contingency_tables["psal"].height == 19480

        assert isinstance(ds.contingency_tables["pres"], pl.DataFrame)
        assert ds.contingency_tables["pres"].height == 19480


class TestKNN:
    """Tests for the K-Nearest Neighbors model wrapper."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test environment and load input, summary, select, and locate data."""
        _setup_classify_all(self)
        model_path = Path(__file__).resolve().parent / "data" / "training"
        self.model_file_names = {
            "temp": str(model_path / "model_temp_knn.joblib"),
            "psal": str(model_path / "model_psal_knn.joblib"),
            "pres": str(model_path / "model_pres_knn.joblib"),
        }

        for x in self.configs:
            x.data["step_class_set"]["steps"]["model"] = "KNearestNeighbors"

    @pytest.mark.parametrize("idx", range(METHOD_TEST_COUNT))
    def test_read_models(self, idx):
        """Confirm that reading models populates the 'models' dictionary with KNearestNeighbors instances."""
        ds = ClassifyAll(
            self.configs[idx],
            test_sets=self.extracts[idx].target_features,
        )
        ds.model_file_names = self.model_file_names
        ds.read_models()

        assert isinstance(ds.models["temp"], KNearestNeighbors)
        assert isinstance(ds.models["psal"], KNearestNeighbors)
        assert isinstance(ds.models["pres"], KNearestNeighbors)

    @pytest.mark.parametrize("idx", range(METHOD_TEST_COUNT))
    def test_with_knn(self, idx):
        """Check that testing targets after model loading populates the result columns and contingency tables."""
        ds = ClassifyAll(
            self.configs[idx],
            test_sets=self.extracts[idx].target_features,
        )
        ds.model_file_names = self.model_file_names
        ds.read_models()
        ds.test_targets()

        # Check Test Sets
        assert isinstance(ds.test_sets["temp"], pl.DataFrame)
        assert ds.test_sets["temp"].shape[0] == 19480
        assert ds.test_sets["temp"].shape[1] == 56

        assert isinstance(ds.test_sets["psal"], pl.DataFrame)
        assert ds.test_sets["psal"].shape[0] == 19480
        assert ds.test_sets["psal"].shape[1] == 56

        assert isinstance(ds.test_sets["pres"], pl.DataFrame)
        assert ds.test_sets["pres"].shape[0] == 19480
        assert ds.test_sets["pres"].shape[1] == 56

        # Check Contingency Tables
        assert isinstance(ds.contingency_tables["temp"], pl.DataFrame)
        assert ds.contingency_tables["temp"].height == 19480
        assert ds.contingency_tables["temp"].columns == [
            "k",
            "label",
            "predicted_label",
            "score",
        ]

        assert isinstance(ds.contingency_tables["psal"], pl.DataFrame)
        assert ds.contingency_tables["psal"].height == 19480

        assert isinstance(ds.contingency_tables["pres"], pl.DataFrame)
        assert ds.contingency_tables["pres"].height == 19480


class TestGaussianNaiveBayes:
    """Tests for the Gaussian Naive Bayes model wrapper."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test environment and load input, summary, select, and locate data."""
        _setup_classify_all(self)
        model_path = Path(__file__).resolve().parent / "data" / "training"
        self.model_file_names = {
            "temp": str(model_path / "model_temp_gnb.joblib"),
            "psal": str(model_path / "model_psal_gnb.joblib"),
            "pres": str(model_path / "model_pres_gnb.joblib"),
        }

        for x in self.configs:
            x.data["step_class_set"]["steps"]["model"] = "GaussianNaiveBayes"

    @pytest.mark.parametrize("idx", range(METHOD_TEST_COUNT))
    def test_read_models(self, idx):
        """Confirm that reading models populates the 'models' dictionary with GaussianNaiveBayes instances."""
        ds = ClassifyAll(
            self.configs[idx],
            test_sets=self.extracts[idx].target_features,
        )
        ds.model_file_names = self.model_file_names
        ds.read_models()

        assert isinstance(ds.models["temp"], GaussianNaiveBayes)
        assert isinstance(ds.models["psal"], GaussianNaiveBayes)
        assert isinstance(ds.models["pres"], GaussianNaiveBayes)

    @pytest.mark.parametrize("idx", range(METHOD_TEST_COUNT))
    def test_with_gnb(self, idx):
        """Check that testing targets after model loading populates the result columns and contingency tables."""
        ds = ClassifyAll(
            self.configs[idx],
            test_sets=self.extracts[idx].target_features,
        )
        ds.model_file_names = self.model_file_names
        ds.read_models()
        ds.test_targets()

        # Check Test Sets
        assert isinstance(ds.test_sets["temp"], pl.DataFrame)
        assert ds.test_sets["temp"].shape[0] == 19480
        assert ds.test_sets["temp"].shape[1] == 56

        assert isinstance(ds.test_sets["psal"], pl.DataFrame)
        assert ds.test_sets["psal"].shape[0] == 19480
        assert ds.test_sets["psal"].shape[1] == 56

        assert isinstance(ds.test_sets["pres"], pl.DataFrame)
        assert ds.test_sets["pres"].shape[0] == 19480
        assert ds.test_sets["pres"].shape[1] == 56

        # Check Contingency Tables
        assert isinstance(ds.contingency_tables["temp"], pl.DataFrame)
        assert ds.contingency_tables["temp"].height == 19480
        assert ds.contingency_tables["temp"].columns == [
            "k",
            "label",
            "predicted_label",
            "score",
        ]

        assert isinstance(ds.contingency_tables["psal"], pl.DataFrame)
        assert ds.contingency_tables["psal"].height == 19480

        assert isinstance(ds.contingency_tables["pres"], pl.DataFrame)
        assert ds.contingency_tables["pres"].height == 19480


class TestMLP:
    """Tests for the Multi-layer Perceptron model wrapper."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test environment and load input, summary, select, and locate data."""
        _setup_classify_all(self)
        model_path = Path(__file__).resolve().parent / "data" / "training"
        self.model_file_names = {
            "temp": str(model_path / "model_temp_mlp.joblib"),
            "psal": str(model_path / "model_psal_mlp.joblib"),
            "pres": str(model_path / "model_pres_mlp.joblib"),
        }

        for x in self.configs:
            x.data["step_class_set"]["steps"]["model"] = "MLP"

    @pytest.mark.parametrize("idx", range(METHOD_TEST_COUNT))
    def test_read_models(self, idx):
        """Confirm that reading models populates the 'models' dictionary with MLP instances."""
        ds = ClassifyAll(
            self.configs[idx],
            test_sets=self.extracts[idx].target_features,
        )
        ds.model_file_names = self.model_file_names
        ds.read_models()

        assert isinstance(ds.models["temp"], MultilayerPerceptron)
        assert isinstance(ds.models["psal"], MultilayerPerceptron)
        assert isinstance(ds.models["pres"], MultilayerPerceptron)

    @pytest.mark.parametrize("idx", range(METHOD_TEST_COUNT))
    def test_with_mlp(self, idx):
        """Check that testing targets after model loading populates the result columns and contingency tables."""
        ds = ClassifyAll(
            self.configs[idx],
            test_sets=self.extracts[idx].target_features,
        )
        ds.model_file_names = self.model_file_names
        ds.read_models()
        ds.test_targets()

        # Check Test Sets
        assert isinstance(ds.test_sets["temp"], pl.DataFrame)
        assert ds.test_sets["temp"].shape[0] == 19480
        assert ds.test_sets["temp"].shape[1] == 56

        assert isinstance(ds.test_sets["psal"], pl.DataFrame)
        assert ds.test_sets["psal"].shape[0] == 19480
        assert ds.test_sets["psal"].shape[1] == 56

        assert isinstance(ds.test_sets["pres"], pl.DataFrame)
        assert ds.test_sets["pres"].shape[0] == 19480
        assert ds.test_sets["pres"].shape[1] == 56

        # Check Contingency Tables
        assert isinstance(ds.contingency_tables["temp"], pl.DataFrame)
        assert ds.contingency_tables["temp"].height == 19480
        assert ds.contingency_tables["temp"].columns == [
            "k",
            "label",
            "predicted_label",
            "score",
        ]

        assert isinstance(ds.contingency_tables["psal"], pl.DataFrame)
        assert ds.contingency_tables["psal"].height == 19480

        assert isinstance(ds.contingency_tables["pres"], pl.DataFrame)
        assert ds.contingency_tables["pres"].height == 19480
