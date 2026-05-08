"""
This module contains unit tests for the BuildModel class, which is responsible
for building, testing, and saving machine learning models, specifically XGBoost models,
within the aiqclib training pipeline.
"""

import os
import unittest
from pathlib import Path

import polars as pl

from aiqclib.common.config.training_config import TrainingConfig
from aiqclib.common.loader.training_loader import load_step1_input_training_set
from aiqclib.train.step4_build_model.build_model import BuildModel
from aiqclib.train.models.logistic_regression import LogisticRegression
from aiqclib.train.models.linear_discriminant_analysis import LinearDiscriminantAnalysis
from aiqclib.train.models.support_vector_machine import SupportVectorMachine
from aiqclib.train.models.decision_tree import DecisionTree
from aiqclib.train.models.random_forest import RandomForest
from aiqclib.train.models.xgboost import XGBoost
from aiqclib.train.models.k_nearest_neighbors import KNearestNeighbors
from aiqclib.train.models.gaussian_naive_bayes import GaussianNaiveBayes
from aiqclib.train.models.multilayer_perceptron import MultilayerPerceptron
from aiqclib.train.step2_validate_model.kfold_validation import KFoldValidation


def setup_training_step4(test_obj):
    """
    Prepare a test training configuration and load input data for subsequent tests.
    Define mock train/test file paths for data loading.
    """
    test_obj.config_file_path = (
        Path(__file__).resolve().parent / "data" / "config" / "test_training_001.yaml"
    )
    test_obj.config = TrainingConfig(str(test_obj.config_file_path))
    test_obj.config.select("NRT_BO_001")
    data_path = Path(__file__).resolve().parent / "data" / "training"
    test_obj.input_file_names = {
        "train": {
            "temp": str(data_path / "train_set_temp.parquet"),
            "psal": str(data_path / "train_set_psal.parquet"),
            "pres": str(data_path / "train_set_pres.parquet"),
        },
        "test": {
            "temp": str(data_path / "test_set_temp.parquet"),
            "psal": str(data_path / "test_set_psal.parquet"),
            "pres": str(data_path / "test_set_pres.parquet"),
        },
    }

    test_obj.ds_input = load_step1_input_training_set(test_obj.config)
    test_obj.ds_input.input_file_names = test_obj.input_file_names
    test_obj.ds_input.process_targets()


def run_test_with_trained_model(test_obj):
    """
    Check that testing sets after model building populates the result columns
    and contingency tables, verifying data types and dimensions remain consistent.
    """
    ds = BuildModel(
        test_obj.config,
        training_sets=test_obj.ds_input.training_sets,
        test_sets=test_obj.ds_input.test_sets,
    )
    ds.build_targets()
    ds.test_targets()

    # Check test sets / predictions
    test_obj.assertIsInstance(ds.test_sets["temp"], pl.DataFrame)
    test_obj.assertEqual(ds.test_sets["temp"].shape[0], 12)
    test_obj.assertEqual(ds.test_sets["temp"].shape[1], 56)

    test_obj.assertIsInstance(ds.test_sets["psal"], pl.DataFrame)
    test_obj.assertEqual(ds.test_sets["psal"].shape[0], 14)
    test_obj.assertEqual(ds.test_sets["psal"].shape[1], 56)

    test_obj.assertIsInstance(ds.test_sets["pres"], pl.DataFrame)
    test_obj.assertEqual(ds.test_sets["pres"].shape[0], 12)
    test_obj.assertEqual(ds.test_sets["pres"].shape[1], 56)

    # Check contingency tables
    test_obj.assertIsInstance(ds.contingency_tables["temp"], pl.DataFrame)
    # Height should match number of test rows
    test_obj.assertEqual(ds.contingency_tables["temp"].height, 12)
    test_obj.assertListEqual(
        ds.contingency_tables["temp"].columns,
        ["k", "label", "predicted_label", "score"],
    )

    test_obj.assertIsInstance(ds.contingency_tables["psal"], pl.DataFrame)
    test_obj.assertEqual(ds.contingency_tables["psal"].height, 14)

    test_obj.assertIsInstance(ds.contingency_tables["pres"], pl.DataFrame)
    test_obj.assertEqual(ds.contingency_tables["pres"].height, 12)


class TestBuildModel(unittest.TestCase):
    """
    A suite of tests ensuring that building, testing, and saving XGBoost models
    via BuildModel follows the expected configuration and data flows.
    """

    def setUp(self):
        """
        Prepare a test training configuration and load input data for subsequent tests.
        Define mock train/test file paths for data loading.
        """
        setup_training_step4(self)

    def test_step_name(self):
        """Check that the BuildModel step name is correctly assigned."""
        ds = BuildModel(self.config)
        self.assertEqual(ds.step_name, "build")

    def test_output_file_names(self):
        """
        Verify that default output file names (model and results) are as expected
        based on the configuration.
        """
        ds = BuildModel(self.config)

        self.assertEqual(
            "/path/to/model_1/nrt_bo_001/model_folder_1/model_temp.joblib",
            str(ds.model_file_names["temp"]),
        )
        self.assertEqual(
            "/path/to/model_1/nrt_bo_001/model_folder_1/model_psal.joblib",
            str(ds.model_file_names["psal"]),
        )
        self.assertEqual(
            "/path/to/model_1/nrt_bo_001/model_folder_1/model_pres.joblib",
            str(ds.model_file_names["pres"]),
        )

        # Check Report paths
        self.assertEqual(
            "/path/to/build_1/nrt_bo_001/build_folder_1/test_report_temp.tsv",
            str(ds.output_file_names["report"]["temp"]),
        )
        self.assertEqual(
            "/path/to/build_1/nrt_bo_001/build_folder_1/test_report_psal.tsv",
            str(ds.output_file_names["report"]["psal"]),
        )
        self.assertEqual(
            "/path/to/build_1/nrt_bo_001/build_folder_1/test_report_pres.tsv",
            str(ds.output_file_names["report"]["pres"]),
        )

        # Check Contingency Table paths
        self.assertEqual(
            "/path/to/build_1/nrt_bo_001/build_folder_1/test_contingency_tables_temp.parquet",
            str(ds.output_file_names["contingency_table"]["temp"]),
        )
        self.assertEqual(
            "/path/to/build_1/nrt_bo_001/build_folder_1/test_contingency_tables_psal.parquet",
            str(ds.output_file_names["contingency_table"]["psal"]),
        )
        self.assertEqual(
            "/path/to/build_1/nrt_bo_001/build_folder_1/test_contingency_tables_pres.parquet",
            str(ds.output_file_names["contingency_table"]["pres"]),
        )

        # Check SHAP value file names
        self.assertEqual(
            "/path/to/build_1/nrt_bo_001/build_folder_1/test_shap_values_temp.parquet",
            str(ds.output_file_names["shap_value"]["temp"]),
        )
        self.assertEqual(
            "/path/to/build_1/nrt_bo_001/build_folder_1/test_shap_values_psal.parquet",
            str(ds.output_file_names["shap_value"]["psal"]),
        )
        self.assertEqual(
            "/path/to/build_1/nrt_bo_001/build_folder_1/test_shap_values_pres.parquet",
            str(ds.output_file_names["shap_value"]["pres"]),
        )

        # Check metric plot file names
        self.assertEqual(
            "/path/to/build_1/nrt_bo_001/build_folder_1/test_metric_plots_temp.svg",
            str(ds.output_file_names["metric_plot"]["temp"]),
        )
        self.assertEqual(
            "/path/to/build_1/nrt_bo_001/build_folder_1/test_metric_plots_psal.svg",
            str(ds.output_file_names["metric_plot"]["psal"]),
        )
        self.assertEqual(
            "/path/to/build_1/nrt_bo_001/build_folder_1/test_metric_plots_pres.svg",
            str(ds.output_file_names["metric_plot"]["pres"]),
        )

    def test_training_sets(self):
        """
        Check that training and test sets are loaded into BuildModel correctly,
        verifying their types and dimensions.
        """
        ds = BuildModel(
            self.config,
            training_sets=self.ds_input.training_sets,
            test_sets=self.ds_input.test_sets,
        )

        self.assertIsInstance(ds.training_sets["temp"], pl.DataFrame)
        self.assertEqual(ds.training_sets["temp"].shape[0], 116)
        self.assertEqual(ds.training_sets["temp"].shape[1], 57)

        self.assertIsInstance(ds.training_sets["psal"], pl.DataFrame)
        self.assertEqual(ds.training_sets["psal"].shape[0], 126)
        self.assertEqual(ds.training_sets["psal"].shape[1], 57)

        self.assertIsInstance(ds.training_sets["pres"], pl.DataFrame)
        self.assertEqual(ds.training_sets["pres"].shape[0], 110)
        self.assertEqual(ds.training_sets["pres"].shape[1], 57)

        self.assertIsInstance(ds.test_sets["temp"], pl.DataFrame)
        self.assertEqual(ds.test_sets["temp"].shape[0], 12)
        self.assertEqual(ds.test_sets["temp"].shape[1], 56)

        self.assertIsInstance(ds.test_sets["psal"], pl.DataFrame)
        self.assertEqual(ds.test_sets["psal"].shape[0], 14)
        self.assertEqual(ds.test_sets["psal"].shape[1], 56)

        self.assertIsInstance(ds.test_sets["pres"], pl.DataFrame)
        self.assertEqual(ds.test_sets["pres"].shape[0], 12)
        self.assertEqual(ds.test_sets["pres"].shape[1], 56)

    def test_shap_flag(self):
        ds = BuildModel(self.config)
        model = ds.base_model
        self.assertFalse(model.enable_shap)

        self.config.data["step_param_set"]["steps"]["model"]["calculate_shap"] = True
        ds = BuildModel(self.config)
        model = ds.base_model
        self.assertTrue(model.enable_shap)

        self.config.data["step_param_set"]["steps"]["model"]["calculate_shap"] = False
        ds = BuildModel(self.config)
        model = ds.base_model
        self.assertFalse(model.enable_shap)

    def test_train_with_xgboost(self):
        """Confirm that building models populates the 'models' dictionary with XGBoost instances."""
        ds = BuildModel(
            self.config,
            training_sets=self.ds_input.training_sets,
            test_sets=self.ds_input.test_sets,
        )
        ds.build_targets()

        self.assertIsInstance(ds.models["temp"], XGBoost)
        self.assertIsInstance(ds.models["psal"], XGBoost)
        self.assertIsInstance(ds.models["pres"], XGBoost)

    def test_train_final_model_with_xgboost(self):
        """Confirm that building test models populates the 'final_models' dictionary with XGBoost instances."""
        ds = BuildModel(
            self.config,
            training_sets=self.ds_input.training_sets,
            test_sets=self.ds_input.test_sets,
        )
        ds.build_final_model_targets()

        self.assertIsInstance(ds.final_models["temp"], XGBoost)
        self.assertIsInstance(ds.final_models["psal"], XGBoost)
        self.assertIsInstance(ds.final_models["pres"], XGBoost)

    def test_model_objects(self):
        """
        Confirm that building final models populates a unique model object for each target.
        Ensures distinct model instances are created, not just references to the same object.
        """
        ds = BuildModel(
            self.config,
            training_sets=self.ds_input.training_sets,
            test_sets=self.ds_input.test_sets,
        )
        ds.build_final_model_targets()

        self.assertIsNot(ds.final_models["temp"], ds.final_models["psal"])
        self.assertIsNot(ds.final_models["temp"], ds.final_models["pres"])
        self.assertIsNot(ds.final_models["psal"], ds.final_models["pres"])

        # Note: assertNotEqual may depend on XGBoost's __eq__ implementation,
        # but assertIsNot is a stronger check for distinct instances.
        self.assertNotEqual(ds.final_models["temp"], ds.final_models["psal"])
        self.assertNotEqual(ds.final_models["temp"], ds.final_models["pres"])
        self.assertNotEqual(ds.final_models["psal"], ds.final_models["pres"])

    def test_test_model_objects(self):
        """
        Confirm that building test models populates a unique model object for each target.
        Ensures distinct model instances are created, not just references to the same object.
        """
        ds = BuildModel(
            self.config,
            training_sets=self.ds_input.training_sets,
            test_sets=self.ds_input.test_sets,
        )
        ds.build_targets()

        self.assertIsNot(ds.models["temp"], ds.models["psal"])
        self.assertIsNot(ds.models["temp"], ds.models["pres"])
        self.assertIsNot(ds.models["psal"], ds.models["pres"])

        # Note: assertNotEqual may depend on XGBoost's __eq__ implementation,
        # but assertIsNot is a stronger check for distinct instances.
        self.assertNotEqual(ds.models["temp"], ds.models["psal"])
        self.assertNotEqual(ds.models["temp"], ds.models["pres"])
        self.assertNotEqual(ds.models["psal"], ds.models["pres"])

    def test_build_without_training_sets(self):
        """Ensure that calling build_targets() with no training sets available raises a ValueError."""
        ds = BuildModel(
            self.config,
            training_sets=None,
            test_sets=None,
        )
        with self.assertRaises(ValueError):
            ds.build_targets()

    def test_build_final_model_without_test_sets(self):
        """Ensure that calling build_final_model_targets() with no training sets available raises a ValueError."""
        ds = BuildModel(
            self.config,
            training_sets=self.ds_input.training_sets,
            test_sets=None,
        )
        with self.assertRaises(ValueError):
            ds.build_final_model_targets()

    def test_build_final_model_without_training_sets(self):
        """Ensure that calling build_final_model_targets() with no training sets available raises a ValueError."""
        ds = BuildModel(
            self.config,
            training_sets=None,
            test_sets=None,
        )
        with self.assertRaises(ValueError):
            ds.build_final_model_targets()

    def test_test_without_model(self):
        """Ensure that calling test_targets() without first building models raises a ValueError."""
        ds = BuildModel(
            self.config,
            training_sets=self.ds_input.training_sets,
            test_sets=self.ds_input.test_sets,
        )
        with self.assertRaises(ValueError):
            ds.test_targets()

    def test_write_reports(self):
        """
        Check that test reports are correctly written to file,
        and then remove the temporary files created.
        """
        ds = BuildModel(
            self.config,
            training_sets=self.ds_input.training_sets,
            test_sets=self.ds_input.test_sets,
        )
        data_path = Path(__file__).resolve().parent / "data" / "training"
        ds.output_file_names["report"]["temp"] = str(
            data_path / "temp_test_report_temp.tsv"
        )
        ds.output_file_names["report"]["psal"] = str(
            data_path / "temp_test_report_psal.tsv"
        )
        ds.output_file_names["report"]["pres"] = str(
            data_path / "temp_test_report_pres.tsv"
        )

        ds.build_targets()
        ds.test_targets()
        ds.write_reports()

        self.assertTrue(os.path.exists(ds.output_file_names["report"]["temp"]))
        self.assertTrue(os.path.exists(ds.output_file_names["report"]["psal"]))
        self.assertTrue(os.path.exists(ds.output_file_names["report"]["pres"]))

        os.remove(ds.output_file_names["report"]["temp"])
        os.remove(ds.output_file_names["report"]["psal"])
        os.remove(ds.output_file_names["report"]["pres"])

    def test_write_contingency_tables(self):
        """
        Check that contingency tables are correctly written to file,
        and then remove the temporary files created.
        """
        ds = BuildModel(
            self.config,
            training_sets=self.ds_input.training_sets,
            test_sets=self.ds_input.test_sets,
        )
        data_path = Path(__file__).resolve().parent / "data" / "training"

        # Override output file names for testing
        ds.output_file_names["contingency_table"]["temp"] = str(
            data_path / "temp_test_contingency_tables_temp.parquet"
        )
        ds.output_file_names["contingency_table"]["psal"] = str(
            data_path / "temp_test_contingency_tables_psal.parquet"
        )
        ds.output_file_names["contingency_table"]["pres"] = str(
            data_path / "temp_test_contingency_tables_pres.parquet"
        )

        ds.build_targets()
        ds.test_targets()
        ds.write_contingency_tables()

        self.assertTrue(
            os.path.exists(ds.output_file_names["contingency_table"]["temp"])
        )
        self.assertTrue(
            os.path.exists(ds.output_file_names["contingency_table"]["psal"])
        )
        self.assertTrue(
            os.path.exists(ds.output_file_names["contingency_table"]["pres"])
        )

        os.remove(ds.output_file_names["contingency_table"]["temp"])
        os.remove(ds.output_file_names["contingency_table"]["psal"])
        os.remove(ds.output_file_names["contingency_table"]["pres"])

    def test_write_shap_values(self):
        """
        Check that contingency tables are correctly written to file,
        and then remove the temporary files created.
        """
        self.config.data["step_param_set"]["steps"]["model"]["calculate_shap"] = True
        ds = BuildModel(
            self.config,
            training_sets=self.ds_input.training_sets,
            test_sets=self.ds_input.test_sets,
        )
        data_path = Path(__file__).resolve().parent / "data" / "training"

        # Override output file names for testing
        ds.output_file_names["shap_value"]["temp"] = str(
            data_path / "temp_test_shap_values_temp.parquet"
        )
        ds.output_file_names["shap_value"]["psal"] = str(
            data_path / "temp_test_shap_values_psal.parquet"
        )
        ds.output_file_names["shap_value"]["pres"] = str(
            data_path / "temp_test_shap_values_pres.parquet"
        )

        ds.build_targets()
        ds.test_targets()
        ds.write_shap_values()

        self.assertTrue(os.path.exists(ds.output_file_names["shap_value"]["temp"]))
        self.assertTrue(os.path.exists(ds.output_file_names["shap_value"]["psal"]))
        self.assertTrue(os.path.exists(ds.output_file_names["shap_value"]["pres"]))

        os.remove(ds.output_file_names["shap_value"]["temp"])
        os.remove(ds.output_file_names["shap_value"]["psal"])
        os.remove(ds.output_file_names["shap_value"]["pres"])

    def test_create_metric_plots(self):
        """
        Ensure ROC and Precision-Recall plots written to the specified output files
        and that these files are created on the file system.
        """
        ds = BuildModel(
            self.config,
            training_sets=self.ds_input.training_sets,
            test_sets=self.ds_input.test_sets,
        )
        data_path = Path(__file__).resolve().parent / "data" / "training"

        # Override output file names for testing
        ds.output_file_names["metric_plot"]["temp"] = str(
            data_path / "temp_test_metric_plots_temp.svg"
        )
        ds.output_file_names["metric_plot"]["psal"] = str(
            data_path / "temp_test_metric_plots_psal.svg"
        )
        ds.output_file_names["metric_plot"]["pres"] = str(
            data_path / "temp_test_metric_plots_pres.svg"
        )

        ds.build_targets()
        ds.test_targets()
        ds.create_metric_plots()

        self.assertTrue(os.path.exists(ds.output_file_names["metric_plot"]["temp"]))
        self.assertTrue(os.path.exists(ds.output_file_names["metric_plot"]["psal"]))
        self.assertTrue(os.path.exists(ds.output_file_names["metric_plot"]["pres"]))

        os.remove(ds.output_file_names["metric_plot"]["temp"])
        os.remove(ds.output_file_names["metric_plot"]["psal"])
        os.remove(ds.output_file_names["metric_plot"]["pres"])

    def test_write_no_results(self):
        """
        Ensure that ValueError is raised if write_reports is called
        with no test results available.
        """
        ds = BuildModel(
            self.config,
            training_sets=self.ds_input.training_sets,
            test_sets=self.ds_input.test_sets,
        )
        with self.assertRaises(ValueError):
            ds.write_reports()

    def test_write_empty_contingency_tables(self):
        """
        Ensure that ValueError is raised if write_contingency_tables is called
        before any tables are generated (e.g. before test_targets).
        """
        ds = BuildModel(
            self.config,
            training_sets=self.ds_input.training_sets,
            test_sets=self.ds_input.test_sets,
        )
        with self.assertRaises(ValueError):
            ds.write_contingency_tables()

    def test_create_empty_metric_plots(self):
        """
        Ensure that ValueError is raised if create_metric_plots is called
        before any tables are generated (e.g. before test_targets).
        """
        ds = BuildModel(
            self.config,
            training_sets=self.ds_input.training_sets,
            test_sets=self.ds_input.test_sets,
        )
        with self.assertRaises(ValueError):
            ds.create_metric_plots()

    def test_write_no_models(self):
        """
        Ensure ValueError is raised if write_models is called without any built models.
        """
        ds = BuildModel(
            self.config,
            training_sets=self.ds_input.training_sets,
            test_sets=self.ds_input.test_sets,
        )
        with self.assertRaises(ValueError):
            ds.write_models()

    def test_write_models(self):
        """
        Check that the trained models are serialized to files correctly,
        and then remove the temporary files created.
        """
        ds = BuildModel(
            self.config,
            training_sets=self.ds_input.training_sets,
            test_sets=self.ds_input.test_sets,
        )
        data_path = Path(__file__).resolve().parent / "data" / "training"
        ds.model_file_names["temp"] = str(data_path / "temp_model_temp.joblib")
        ds.model_file_names["psal"] = str(data_path / "temp_model_psal.joblib")
        ds.model_file_names["pres"] = str(data_path / "temp_model_pres.joblib")

        ds.build_final_model_targets()
        ds.write_models()

        self.assertTrue(os.path.exists(ds.model_file_names["temp"]))
        self.assertTrue(os.path.exists(ds.model_file_names["psal"]))
        self.assertTrue(os.path.exists(ds.model_file_names["pres"]))

        os.remove(ds.model_file_names["temp"])
        os.remove(ds.model_file_names["psal"])
        os.remove(ds.model_file_names["pres"])

    def test_read_models(self):
        """
        Verify that existing models can be reloaded from disk and successfully
        used for testing.
        """
        ds = BuildModel(
            self.config, training_sets=None, test_sets=self.ds_input.test_sets
        )
        data_path = Path(__file__).resolve().parent / "data" / "training"
        ds.model_file_names["temp"] = str(data_path / "model_temp.joblib")
        ds.model_file_names["psal"] = str(data_path / "model_psal.joblib")
        ds.model_file_names["pres"] = str(data_path / "model_pres.joblib")

        ds.read_models()

        self.assertIsInstance(ds.models["temp"], XGBoost)
        self.assertIsInstance(ds.models["psal"], XGBoost)
        self.assertIsInstance(ds.models["pres"], XGBoost)

        ds.test_targets()

        self.assertIsInstance(ds.test_sets["temp"], pl.DataFrame)
        self.assertEqual(ds.test_sets["temp"].shape[0], 12)
        self.assertEqual(ds.test_sets["temp"].shape[1], 56)

        self.assertIsInstance(ds.test_sets["psal"], pl.DataFrame)
        self.assertEqual(ds.test_sets["psal"].shape[0], 14)
        self.assertEqual(ds.test_sets["psal"].shape[1], 56)

    def test_read_models_no_file(self):
        """Check that FileNotFoundError is raised if model files are missing during reading."""
        ds = BuildModel(
            self.config, training_sets=None, test_sets=self.ds_input.test_sets
        )
        data_path = Path(__file__).resolve().parent / "data" / "training"
        ds.model_file_names["temp"] = str(data_path / "non_existent_model.joblib")
        ds.model_file_names["psal"] = str(data_path / "non_existent_model.joblib")
        ds.model_file_names["pres"] = str(data_path / "non_existent_model.joblib")

        with self.assertRaises(FileNotFoundError):
            ds.read_models()

    def test_write_predictions(self):
        """
        Check that test predictions are correctly written to file,
        and then remove the temporary files created.
        """
        ds = BuildModel(
            self.config,
            training_sets=self.ds_input.training_sets,
            test_sets=self.ds_input.test_sets,
        )
        data_path = Path(__file__).resolve().parent / "data" / "training"
        ds.output_file_names["prediction"]["temp"] = str(
            data_path / "temp_test_prediction_temp.parquet"
        )
        ds.output_file_names["prediction"]["psal"] = str(
            data_path / "temp_test_prediction_psal.parquet"
        )
        ds.output_file_names["prediction"]["pres"] = str(
            data_path / "temp_test_prediction_pres.parquet"
        )

        ds.build_targets()
        ds.test_targets()
        ds.write_predictions()

        self.assertTrue(os.path.exists(ds.output_file_names["prediction"]["temp"]))
        self.assertTrue(os.path.exists(ds.output_file_names["prediction"]["psal"]))
        self.assertTrue(os.path.exists(ds.output_file_names["prediction"]["pres"]))

        os.remove(ds.output_file_names["prediction"]["temp"])
        os.remove(ds.output_file_names["prediction"]["psal"])
        os.remove(ds.output_file_names["prediction"]["pres"])

    def test_write_empty_predictions(self):
        """
        Ensure that calling write_predictions() before predictions are generated
        (i.e., before test_targets() is called) raises a ValueError.
        """
        ds = BuildModel(
            self.config,
            training_sets=self.ds_input.training_sets,
            test_sets=self.ds_input.test_sets,
        )
        with self.assertRaises(ValueError):
            ds.write_predictions()


class TestXGBoost(unittest.TestCase):
    """Tests for the XGBoost model wrapper."""

    def setUp(self):
        """
        Prepare the test environment by loading a training configuration
        and input training data. The input file names for train/test sets
        are defined here for subsequent model validation tests.
        """
        setup_training_step4(self)

        self.config.data["step_class_set"]["steps"]["model"] = "XGBoost"

    def test_base_model(self):
        """
        Ensure the base model attribute of KFoldValidation is an XGBoost
        instance, as defined by the configuration.
        """
        ds = KFoldValidation(self.config)
        self.assertIsInstance(ds.base_model, XGBoost)

    def test_trained_model(self):
        """Confirm that building models populates the 'models' dictionary with XGBoost instances."""
        ds = BuildModel(
            self.config,
            training_sets=self.ds_input.training_sets,
            test_sets=self.ds_input.test_sets,
        )
        ds.build_targets()

        self.assertIsInstance(ds.models["temp"], XGBoost)
        self.assertIsInstance(ds.models["psal"], XGBoost)
        self.assertIsInstance(ds.models["pres"], XGBoost)

    def test_model_output(self):
        """
        Check that testing sets after model building populates the result columns
        and contingency tables, verifying data types and dimensions remain consistent.
        """
        run_test_with_trained_model(self)

    def test_write_model(self):
        ds = BuildModel(
            self.config,
            training_sets=self.ds_input.training_sets,
            test_sets=self.ds_input.test_sets,
        )
        data_path = Path(__file__).resolve().parent / "data" / "training"
        ds.model_file_names["temp"] = str(data_path / "temp_model_temp_xgboost.joblib")
        ds.model_file_names["psal"] = str(data_path / "temp_model_psal_xgboost.joblib")
        ds.model_file_names["pres"] = str(data_path / "temp_model_pres_xgboost.joblib")

        ds.build_final_model_targets()
        ds.write_models()

        self.assertTrue(os.path.exists(ds.model_file_names["temp"]))
        self.assertTrue(os.path.exists(ds.model_file_names["psal"]))
        self.assertTrue(os.path.exists(ds.model_file_names["pres"]))

        os.remove(ds.model_file_names["temp"])
        os.remove(ds.model_file_names["psal"])
        os.remove(ds.model_file_names["pres"])


class TestLogisticRegression(unittest.TestCase):
    """Tests for the Logistic Regression model wrapper."""

    def setUp(self):
        setup_training_step4(self)

        self.config.data["step_class_set"]["steps"]["model"] = "LogisticRegression"

    def test_base_model(self):
        """
        Ensure the base model attribute of KFoldValidation is a LogisticRegression
        instance, as defined by the configuration.
        """
        ds = KFoldValidation(self.config)
        self.assertIsInstance(ds.base_model, LogisticRegression)

    def test_training(self):
        """Confirm that building models populates the 'models' dictionary with LogisticRegression instances."""
        ds = BuildModel(
            self.config,
            training_sets=self.ds_input.training_sets,
            test_sets=self.ds_input.test_sets,
        )
        ds.build_targets()

        self.assertIsInstance(ds.models["temp"], LogisticRegression)
        self.assertIsInstance(ds.models["psal"], LogisticRegression)
        self.assertIsInstance(ds.models["pres"], LogisticRegression)

    def test_model_output(self):
        """
        Check that testing sets after model building populates the result columns
        and contingency tables, verifying data types and dimensions remain consistent.
        """
        run_test_with_trained_model(self)

    def test_write_model(self):
        ds = BuildModel(
            self.config,
            training_sets=self.ds_input.training_sets,
            test_sets=self.ds_input.test_sets,
        )
        data_path = Path(__file__).resolve().parent / "data" / "training"
        ds.model_file_names["temp"] = str(data_path / "temp_model_temp_logit.joblib")
        ds.model_file_names["psal"] = str(data_path / "temp_model_psal_logit.joblib")
        ds.model_file_names["pres"] = str(data_path / "temp_model_pres_logit.joblib")

        ds.build_final_model_targets()
        ds.write_models()

        self.assertTrue(os.path.exists(ds.model_file_names["temp"]))
        self.assertTrue(os.path.exists(ds.model_file_names["psal"]))
        self.assertTrue(os.path.exists(ds.model_file_names["pres"]))

        os.remove(ds.model_file_names["temp"])
        os.remove(ds.model_file_names["psal"])
        os.remove(ds.model_file_names["pres"])


class TestLDA(unittest.TestCase):
    """Tests for the Linear Discriminant Analysis model wrapper."""

    def setUp(self):
        setup_training_step4(self)

        self.config.data["step_class_set"]["steps"]["model"] = (
            "LinearDiscriminantAnalysis"
        )

    def test_base_model(self):
        """
        Ensure the base model attribute of KFoldValidation is a LinearDiscriminantAnalysis
        instance, as defined by the configuration.
        """
        ds = KFoldValidation(self.config)
        self.assertIsInstance(ds.base_model, LinearDiscriminantAnalysis)

    def test_training(self):
        """Confirm that building models populates the 'models' dictionary with LinearDiscriminantAnalysis instances."""
        ds = BuildModel(
            self.config,
            training_sets=self.ds_input.training_sets,
            test_sets=self.ds_input.test_sets,
        )
        ds.build_targets()

        self.assertIsInstance(ds.models["temp"], LinearDiscriminantAnalysis)
        self.assertIsInstance(ds.models["psal"], LinearDiscriminantAnalysis)
        self.assertIsInstance(ds.models["pres"], LinearDiscriminantAnalysis)

    def test_model_output(self):
        """
        Check that testing sets after model building populates the result columns
        and contingency tables, verifying data types and dimensions remain consistent.
        """
        run_test_with_trained_model(self)

    def test_write_model(self):
        ds = BuildModel(
            self.config,
            training_sets=self.ds_input.training_sets,
            test_sets=self.ds_input.test_sets,
        )
        data_path = Path(__file__).resolve().parent / "data" / "training"
        ds.model_file_names["temp"] = str(data_path / "temp_model_temp_lda.joblib")
        ds.model_file_names["psal"] = str(data_path / "temp_model_psal_lda.joblib")
        ds.model_file_names["pres"] = str(data_path / "temp_model_pres_lda.joblib")

        ds.build_final_model_targets()
        ds.write_models()

        self.assertTrue(os.path.exists(ds.model_file_names["temp"]))
        self.assertTrue(os.path.exists(ds.model_file_names["psal"]))
        self.assertTrue(os.path.exists(ds.model_file_names["pres"]))

        os.remove(ds.model_file_names["temp"])
        os.remove(ds.model_file_names["psal"])
        os.remove(ds.model_file_names["pres"])


class TestSVM(unittest.TestCase):
    """Tests for the SVM model wrapper."""

    def setUp(self):
        setup_training_step4(self)

        self.config.data["step_class_set"]["steps"]["model"] = "SVM"

    def test_base_model(self):
        """
        Ensure the base model attribute of KFoldValidation is an SVM
        instance, as defined by the configuration.
        """
        ds = KFoldValidation(self.config)
        self.assertIsInstance(ds.base_model, SupportVectorMachine)

    def test_training(self):
        """Confirm that building models populates the 'models' dictionary with SVM instances."""
        ds = BuildModel(
            self.config,
            training_sets=self.ds_input.training_sets,
            test_sets=self.ds_input.test_sets,
        )
        ds.build_targets()

        self.assertIsInstance(ds.models["temp"], SupportVectorMachine)
        self.assertIsInstance(ds.models["psal"], SupportVectorMachine)
        self.assertIsInstance(ds.models["pres"], SupportVectorMachine)

    def test_model_output(self):
        """
        Check that testing sets after model building populates the result columns
        and contingency tables, verifying data types and dimensions remain consistent.
        """
        run_test_with_trained_model(self)

    def test_write_model(self):
        ds = BuildModel(
            self.config,
            training_sets=self.ds_input.training_sets,
            test_sets=self.ds_input.test_sets,
        )
        data_path = Path(__file__).resolve().parent / "data" / "training"
        ds.model_file_names["temp"] = str(data_path / "temp_model_temp_svm.joblib")
        ds.model_file_names["psal"] = str(data_path / "temp_model_psal_svm.joblib")
        ds.model_file_names["pres"] = str(data_path / "temp_model_pres_svm.joblib")

        ds.build_final_model_targets()
        ds.write_models()

        self.assertTrue(os.path.exists(ds.model_file_names["temp"]))
        self.assertTrue(os.path.exists(ds.model_file_names["psal"]))
        self.assertTrue(os.path.exists(ds.model_file_names["pres"]))

        os.remove(ds.model_file_names["temp"])
        os.remove(ds.model_file_names["psal"])
        os.remove(ds.model_file_names["pres"])


class TestDecisionTree(unittest.TestCase):
    """Tests for the Decision Tree model wrapper."""

    def setUp(self):
        setup_training_step4(self)

        self.config.data["step_class_set"]["steps"]["model"] = "DecisionTree"

    def test_base_model(self):
        """
        Ensure the base model attribute of KFoldValidation is a DecisionTree
        instance, as defined by the configuration.
        """
        ds = KFoldValidation(self.config)
        self.assertIsInstance(ds.base_model, DecisionTree)

    def test_training(self):
        """Confirm that building models populates the 'models' dictionary with DecisionTree instances."""
        ds = BuildModel(
            self.config,
            training_sets=self.ds_input.training_sets,
            test_sets=self.ds_input.test_sets,
        )
        ds.build_targets()

        self.assertIsInstance(ds.models["temp"], DecisionTree)
        self.assertIsInstance(ds.models["psal"], DecisionTree)
        self.assertIsInstance(ds.models["pres"], DecisionTree)

    def test_model_output(self):
        """
        Check that testing sets after model building populates the result columns
        and contingency tables, verifying data types and dimensions remain consistent.
        """
        run_test_with_trained_model(self)

    def test_write_model(self):
        ds = BuildModel(
            self.config,
            training_sets=self.ds_input.training_sets,
            test_sets=self.ds_input.test_sets,
        )
        data_path = Path(__file__).resolve().parent / "data" / "training"
        ds.model_file_names["temp"] = str(data_path / "temp_model_temp_dt.joblib")
        ds.model_file_names["psal"] = str(data_path / "temp_model_psal_dt.joblib")
        ds.model_file_names["pres"] = str(data_path / "temp_model_pres_dt.joblib")

        ds.build_final_model_targets()
        ds.write_models()

        self.assertTrue(os.path.exists(ds.model_file_names["temp"]))
        self.assertTrue(os.path.exists(ds.model_file_names["psal"]))
        self.assertTrue(os.path.exists(ds.model_file_names["pres"]))

        os.remove(ds.model_file_names["temp"])
        os.remove(ds.model_file_names["psal"])
        os.remove(ds.model_file_names["pres"])


class TestRandomForest(unittest.TestCase):
    """Tests for the Random Forest model wrapper."""

    def setUp(self):
        setup_training_step4(self)

        self.config.data["step_class_set"]["steps"]["model"] = "RandomForest"

    def test_base_model(self):
        """
        Ensure the base model attribute of KFoldValidation is a RandomForest
        instance, as defined by the configuration.
        """
        ds = KFoldValidation(self.config)
        self.assertIsInstance(ds.base_model, RandomForest)

    def test_training(self):
        """Confirm that building models populates the 'models' dictionary with RandomForest instances."""
        ds = BuildModel(
            self.config,
            training_sets=self.ds_input.training_sets,
            test_sets=self.ds_input.test_sets,
        )
        ds.build_targets()

        self.assertIsInstance(ds.models["temp"], RandomForest)
        self.assertIsInstance(ds.models["psal"], RandomForest)
        self.assertIsInstance(ds.models["pres"], RandomForest)

    def test_model_output(self):
        """
        Check that testing sets after model building populates the result columns
        and contingency tables, verifying data types and dimensions remain consistent.
        """
        run_test_with_trained_model(self)

    def test_write_model(self):
        ds = BuildModel(
            self.config,
            training_sets=self.ds_input.training_sets,
            test_sets=self.ds_input.test_sets,
        )
        data_path = Path(__file__).resolve().parent / "data" / "training"
        ds.model_file_names["temp"] = str(data_path / "temp_model_temp_rf.joblib")
        ds.model_file_names["psal"] = str(data_path / "temp_model_psal_rf.joblib")
        ds.model_file_names["pres"] = str(data_path / "temp_model_pres_rf.joblib")

        ds.build_final_model_targets()
        ds.write_models()

        self.assertTrue(os.path.exists(ds.model_file_names["temp"]))
        self.assertTrue(os.path.exists(ds.model_file_names["psal"]))
        self.assertTrue(os.path.exists(ds.model_file_names["pres"]))

        os.remove(ds.model_file_names["temp"])
        os.remove(ds.model_file_names["psal"])
        os.remove(ds.model_file_names["pres"])


class TestKNN(unittest.TestCase):
    """Tests for the K-Nearest Neighbors model wrapper."""

    def setUp(self):
        setup_training_step4(self)

        self.config.data["step_class_set"]["steps"]["model"] = "KNearestNeighbors"

    def test_base_model(self):
        """
        Ensure the base model attribute of KFoldValidation is a KNearestNeighbors
        instance, as defined by the configuration.
        """
        ds = KFoldValidation(self.config)
        self.assertIsInstance(ds.base_model, KNearestNeighbors)

    def test_training(self):
        """Confirm that building models populates the 'models' dictionary with KNearestNeighbors instances."""
        ds = BuildModel(
            self.config,
            training_sets=self.ds_input.training_sets,
            test_sets=self.ds_input.test_sets,
        )
        ds.build_targets()

        self.assertIsInstance(ds.models["temp"], KNearestNeighbors)
        self.assertIsInstance(ds.models["psal"], KNearestNeighbors)
        self.assertIsInstance(ds.models["pres"], KNearestNeighbors)

    def test_model_output(self):
        """
        Check that testing sets after model building populates the result columns
        and contingency tables, verifying data types and dimensions remain consistent.
        """
        run_test_with_trained_model(self)

    def test_write_model(self):
        ds = BuildModel(
            self.config,
            training_sets=self.ds_input.training_sets,
            test_sets=self.ds_input.test_sets,
        )
        data_path = Path(__file__).resolve().parent / "data" / "training"
        ds.model_file_names["temp"] = str(data_path / "temp_model_temp_knn.joblib")
        ds.model_file_names["psal"] = str(data_path / "temp_model_psal_knn.joblib")
        ds.model_file_names["pres"] = str(data_path / "temp_model_pres_knn.joblib")

        ds.build_final_model_targets()
        ds.write_models()

        self.assertTrue(os.path.exists(ds.model_file_names["temp"]))
        self.assertTrue(os.path.exists(ds.model_file_names["psal"]))
        self.assertTrue(os.path.exists(ds.model_file_names["pres"]))

        os.remove(ds.model_file_names["temp"])
        os.remove(ds.model_file_names["psal"])
        os.remove(ds.model_file_names["pres"])


class TestGaussianNaiveBayes(unittest.TestCase):
    """Tests for the Gaussian Naive Bayes model wrapper."""

    def setUp(self):
        setup_training_step4(self)

        self.config.data["step_class_set"]["steps"]["model"] = "GaussianNaiveBayes"

    def test_base_model(self):
        """
        Ensure the base model attribute of KFoldValidation is a GaussianNaiveBayes
        instance, as defined by the configuration.
        """
        ds = KFoldValidation(self.config)
        self.assertIsInstance(ds.base_model, GaussianNaiveBayes)

    def test_training(self):
        """Confirm that building models populates the 'models' dictionary with GaussianNaiveBayes instances."""
        ds = BuildModel(
            self.config,
            training_sets=self.ds_input.training_sets,
            test_sets=self.ds_input.test_sets,
        )
        ds.build_targets()

        self.assertIsInstance(ds.models["temp"], GaussianNaiveBayes)
        self.assertIsInstance(ds.models["psal"], GaussianNaiveBayes)
        self.assertIsInstance(ds.models["pres"], GaussianNaiveBayes)

    def test_model_output(self):
        """
        Check that testing sets after model building populates the result columns
        and contingency tables, verifying data types and dimensions remain consistent.
        """
        run_test_with_trained_model(self)

    def test_write_model(self):
        ds = BuildModel(
            self.config,
            training_sets=self.ds_input.training_sets,
            test_sets=self.ds_input.test_sets,
        )
        data_path = Path(__file__).resolve().parent / "data" / "training"
        ds.model_file_names["temp"] = str(data_path / "temp_model_temp_gnb.joblib")
        ds.model_file_names["psal"] = str(data_path / "temp_model_psal_gnb.joblib")
        ds.model_file_names["pres"] = str(data_path / "temp_model_pres_gnb.joblib")

        ds.build_final_model_targets()
        ds.write_models()

        self.assertTrue(os.path.exists(ds.model_file_names["temp"]))
        self.assertTrue(os.path.exists(ds.model_file_names["psal"]))
        self.assertTrue(os.path.exists(ds.model_file_names["pres"]))

        os.remove(ds.model_file_names["temp"])
        os.remove(ds.model_file_names["psal"])
        os.remove(ds.model_file_names["pres"])


class TestMLP(unittest.TestCase):
    """Tests for the Multi-layer Perceptron model wrapper."""

    def setUp(self):
        setup_training_step4(self)

        self.config.data["step_class_set"]["steps"]["model"] = "MLP"

    def test_base_model(self):
        """
        Ensure the base model attribute of KFoldValidation is a MLP
        instance, as defined by the configuration.
        """
        ds = KFoldValidation(self.config)
        self.assertIsInstance(ds.base_model, MultilayerPerceptron)

    def test_training(self):
        """Confirm that building models populates the 'models' dictionary with MLP instances."""
        ds = BuildModel(
            self.config,
            training_sets=self.ds_input.training_sets,
            test_sets=self.ds_input.test_sets,
        )
        ds.build_targets()

        self.assertIsInstance(ds.models["temp"], MultilayerPerceptron)
        self.assertIsInstance(ds.models["psal"], MultilayerPerceptron)
        self.assertIsInstance(ds.models["pres"], MultilayerPerceptron)

    def test_model_output(self):
        """
        Check that testing sets after model building populates the result columns
        and contingency tables, verifying data types and dimensions remain consistent.
        """
        run_test_with_trained_model(self)

    def test_write_model(self):
        ds = BuildModel(
            self.config,
            training_sets=self.ds_input.training_sets,
            test_sets=self.ds_input.test_sets,
        )
        data_path = Path(__file__).resolve().parent / "data" / "training"
        ds.model_file_names["temp"] = str(data_path / "temp_model_temp_mlp.joblib")
        ds.model_file_names["psal"] = str(data_path / "temp_model_psal_mlp.joblib")
        ds.model_file_names["pres"] = str(data_path / "temp_model_pres_mlp.joblib")

        ds.build_final_model_targets()
        ds.write_models()

        self.assertTrue(os.path.exists(ds.model_file_names["temp"]))
        self.assertTrue(os.path.exists(ds.model_file_names["psal"]))
        self.assertTrue(os.path.exists(ds.model_file_names["pres"]))

        os.remove(ds.model_file_names["temp"])
        os.remove(ds.model_file_names["psal"])
        os.remove(ds.model_file_names["pres"])
