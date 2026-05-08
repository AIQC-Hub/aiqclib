"""
This module contains unit tests for the KFoldValidation class, ensuring its
correct integration with training configurations, data loading, model execution
(XGBoost), and report generation. It verifies that the validation process
behaves as expected across various scenarios.
"""

import os
import unittest
from pathlib import Path

import polars as pl

from aiqclib.common.config.training_config import TrainingConfig
from aiqclib.common.loader.training_loader import load_step1_input_training_set
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


def setup_training_step2(test_obj):
    """
    Prepare the test environment by loading a training configuration
    and input training data. The input file names for train/test sets
    are defined here for subsequent model validation tests.
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


def run_fold_validation(test_obj):
    ds = KFoldValidation(test_obj.config, training_sets=test_obj.ds_input.training_sets)
    ds.process_targets()

    # Check Reports
    test_obj.assertIsInstance(ds.reports["temp"], pl.DataFrame)
    test_obj.assertEqual(ds.reports["temp"].shape[0], 18)
    test_obj.assertEqual(ds.reports["temp"].shape[1], 8)

    # Check Contingency Tables
    # "temp" has 116 rows in training set; K-fold should result in 116 predictions total.
    test_obj.assertIsInstance(ds.contingency_tables["temp"], pl.DataFrame)
    test_obj.assertEqual(ds.contingency_tables["temp"].height, 116)
    # Expected columns: k, label, score
    test_obj.assertListEqual(
        ds.contingency_tables["temp"].columns,
        ["k", "label", "predicted_label", "score"],
    )

    # "psal" has 126 rows
    test_obj.assertIsInstance(ds.contingency_tables["psal"], pl.DataFrame)
    test_obj.assertEqual(ds.contingency_tables["psal"].height, 126)

    # "pres" has 110 rows
    test_obj.assertIsInstance(ds.contingency_tables["pres"], pl.DataFrame)
    test_obj.assertEqual(ds.contingency_tables["pres"].height, 110)


class TestKFoldValidation(unittest.TestCase):
    """
    A suite of tests ensuring that KFoldValidation correctly captures
    configurations, splits training data, applies the XGBoost model,
    and writes validation results.
    """

    def setUp(self):
        """
        Prepare the test environment by loading a training configuration
        and input training data. The input file names for train/test sets
        are defined here for subsequent model validation tests.
        """
        setup_training_step2(self)

    def test_step_name(self):
        """
        Check that the step name is correctly identified as 'validate'.
        """
        ds = KFoldValidation(self.config)
        self.assertEqual(ds.step_name, "validate")

    def test_output_file_names(self):
        """
        Verify that the default output file names are correctly resolved
        based on the configuration.
        """
        ds = KFoldValidation(self.config)
        # Check report file names
        self.assertEqual(
            "/path/to/validate_1/nrt_bo_001/validate_folder_1/validation_report_temp.tsv",
            str(ds.output_file_names["report"]["temp"]),
        )
        self.assertEqual(
            "/path/to/validate_1/nrt_bo_001/validate_folder_1/validation_report_psal.tsv",
            str(ds.output_file_names["report"]["psal"]),
        )

        # Check contingency table file names
        self.assertEqual(
            "/path/to/validate_1/nrt_bo_001/validate_folder_1/contingency_tables_temp.parquet",
            str(ds.output_file_names["contingency_table"]["temp"]),
        )
        self.assertEqual(
            "/path/to/validate_1/nrt_bo_001/validate_folder_1/contingency_tables_psal.parquet",
            str(ds.output_file_names["contingency_table"]["psal"]),
        )

        # Check metric plot file names
        self.assertEqual(
            "/path/to/validate_1/nrt_bo_001/validate_folder_1/metric_plots_temp.svg",
            str(ds.output_file_names["metric_plot"]["temp"]),
        )
        self.assertEqual(
            "/path/to/validate_1/nrt_bo_001/validate_folder_1/metric_plots_psal.svg",
            str(ds.output_file_names["metric_plot"]["psal"]),
        )

    def test_base_model(self):
        """
        Ensure the base model attribute of KFoldValidation is an XGBoost
        instance, as defined by the configuration.
        """
        ds = KFoldValidation(self.config)
        self.assertIsInstance(ds.base_model, XGBoost)

    def test_training_sets(self):
        """
        Check that training data is properly loaded and accessible
        within the KFoldValidation instance.
        """
        ds = KFoldValidation(self.config, training_sets=self.ds_input.training_sets)

        self.assertIsInstance(ds.training_sets["temp"], pl.DataFrame)
        self.assertEqual(ds.training_sets["temp"].shape[0], 116)
        self.assertEqual(ds.training_sets["temp"].shape[1], 57)

        self.assertIsInstance(ds.training_sets["psal"], pl.DataFrame)
        self.assertEqual(ds.training_sets["psal"].shape[0], 126)
        self.assertEqual(ds.training_sets["psal"].shape[1], 57)

        self.assertIsInstance(ds.training_sets["pres"], pl.DataFrame)
        self.assertEqual(ds.training_sets["pres"].shape[0], 110)
        self.assertEqual(ds.training_sets["pres"].shape[1], 57)

    def test_shap_flag(self):
        ds = KFoldValidation(self.config)
        model = ds.base_model
        self.assertFalse(model.enable_shap)

        self.config.data["step_param_set"]["steps"]["model"]["calculate_shap"] = True
        ds = KFoldValidation(self.config)
        model = ds.base_model
        self.assertFalse(model.enable_shap)

        self.config.data["step_param_set"]["steps"]["model"]["calculate_shap"] = False
        ds = KFoldValidation(self.config)
        model = ds.base_model
        self.assertFalse(model.enable_shap)

    def test_default_k_fold(self):
        """
        Confirm that the k_fold value defaults to 10 if no specific
        configuration entry is present for it.
        """
        ds = KFoldValidation(self.config, training_sets=self.ds_input.training_sets)
        # Temporarily modify config data to simulate missing k_fold setting
        ds.config.data["step_param_set"]["steps"]["validate"]["k_fold"] = None

        k_fold = ds.get_k_fold()
        self.assertEqual(k_fold, 10)

    def test_write_results(self):
        """
        Ensure validation reports are written to the specified output files
        and that these files are created on the file system.
        Temporary files are cleaned up after the test.
        """
        ds = KFoldValidation(self.config, training_sets=self.ds_input.training_sets)

        data_path = Path(__file__).resolve().parent / "data" / "training"
        ds.output_file_names["report"]["temp"] = str(
            data_path / "temp_validation_report_temp.tsv"
        )
        ds.output_file_names["report"]["psal"] = str(
            data_path / "temp_validation_report_psal.tsv"
        )
        ds.output_file_names["report"]["pres"] = str(
            data_path / "temp_validation_report_pres.tsv"
        )

        ds.process_targets()
        ds.write_reports()

        self.assertTrue(os.path.exists(ds.output_file_names["report"]["temp"]))
        self.assertTrue(os.path.exists(ds.output_file_names["report"]["psal"]))
        self.assertTrue(os.path.exists(ds.output_file_names["report"]["pres"]))

        os.remove(ds.output_file_names["report"]["temp"])
        os.remove(ds.output_file_names["report"]["psal"])
        os.remove(ds.output_file_names["report"]["pres"])

    def test_write_contingency_tables(self):
        """
        Ensure contingency tables are written to the specified output files
        and that these files are created on the file system.
        """
        ds = KFoldValidation(self.config, training_sets=self.ds_input.training_sets)
        data_path = Path(__file__).resolve().parent / "data" / "training"

        # Override output paths for testing
        ds.output_file_names["contingency_table"]["temp"] = str(
            data_path / "temp_contingency_temp.parquet"
        )
        ds.output_file_names["contingency_table"]["psal"] = str(
            data_path / "temp_contingency_psal.parquet"
        )
        ds.output_file_names["contingency_table"]["pres"] = str(
            data_path / "temp_contingency_pres.parquet"
        )

        ds.process_targets()
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

        # Cleanup
        os.remove(ds.output_file_names["contingency_table"]["temp"])
        os.remove(ds.output_file_names["contingency_table"]["psal"])
        os.remove(ds.output_file_names["contingency_table"]["pres"])

    def test_create_metric_plots(self):
        """
        Ensure ROC and Precision-Recall plots written to the specified output files
        and that these files are created on the file system.
        """
        ds = KFoldValidation(self.config, training_sets=self.ds_input.training_sets)
        data_path = Path(__file__).resolve().parent / "data" / "training"

        # Override output paths for testing
        ds.output_file_names["metric_plot"]["temp"] = str(
            data_path / "temp_metric_plots_temp.svg"
        )
        ds.output_file_names["metric_plot"]["psal"] = str(
            data_path / "temp_metric_plots_psal.svg"
        )
        ds.output_file_names["metric_plot"]["pres"] = str(
            data_path / "temp_metric_plots_pres.svg"
        )

        ds.process_targets()
        ds.create_metric_plots()

        self.assertTrue(os.path.exists(ds.output_file_names["metric_plot"]["temp"]))
        self.assertTrue(os.path.exists(ds.output_file_names["metric_plot"]["psal"]))
        self.assertTrue(os.path.exists(ds.output_file_names["metric_plot"]["pres"]))

        # Cleanup
        os.remove(ds.output_file_names["metric_plot"]["temp"])
        os.remove(ds.output_file_names["metric_plot"]["psal"])
        os.remove(ds.output_file_names["metric_plot"]["pres"])

    def test_write_reports_empty_reports(self):
        """
        Ensure that calling write_reports with empty reports (i.e., before
        process_targets has been called) raises a ValueError.
        """
        ds = KFoldValidation(self.config, training_sets=self.ds_input.training_sets)
        with self.assertRaises(ValueError):
            ds.write_reports()

    def test_write_contingency_tables_empty(self):
        """
        Ensure that calling write_contingency_tables with empty tables (i.e., before
        process_targets has been called) raises a ValueError.
        """
        ds = KFoldValidation(self.config, training_sets=self.ds_input.training_sets)
        with self.assertRaises(ValueError):
            ds.write_contingency_tables()

    def test_create_metric_plots_empty(self):
        """
        Ensure that calling create_metric_plots with empty tables (i.e., before
        process_targets has been called) raises a ValueError.
        """
        ds = KFoldValidation(self.config, training_sets=self.ds_input.training_sets)
        with self.assertRaises(ValueError):
            ds.create_metric_plots()


class TestXGBoost(unittest.TestCase):
    """Tests for the XGBoost model wrapper."""

    def setUp(self):
        """
        Prepare the test environment by loading a training configuration
        and input training data. The input file names for train/test sets
        are defined here for subsequent model validation tests.
        """
        setup_training_step2(self)

        self.config.data["step_class_set"]["steps"]["model"] = "XGBoost"

    def test_base_model(self):
        """
        Ensure the base model attribute of KFoldValidation is an XGBoost
        instance, as defined by the configuration.
        """
        ds = KFoldValidation(self.config)
        self.assertIsInstance(ds.base_model, XGBoost)

    def test_fold_validation(self):
        """
        Check that the KFoldValidation process, utilizing the XGBoost model,
        successfully processes the training sets and populates both the reports
        and the contingency tables via the validate method.
        """
        run_fold_validation(self)


class TestLogisticRegression(unittest.TestCase):
    """Tests for the Logistic Regression model wrapper."""

    def setUp(self):
        setup_training_step2(self)

        self.config.data["step_class_set"]["steps"]["model"] = "LogisticRegression"

    def test_base_model(self):
        """
        Ensure the base model attribute of KFoldValidation is a LogisticRegression
        instance, as defined by the configuration.
        """
        ds = KFoldValidation(self.config)
        self.assertIsInstance(ds.base_model, LogisticRegression)

    def test_fold_validation(self):
        """
        Check that the KFoldValidation process, utilizing the LogisticRegression model,
        successfully processes the training sets and populates both the reports
        and the contingency tables via the validate method.
        """
        run_fold_validation(self)


class TestLDA(unittest.TestCase):
    """Tests for the Linear Discriminant Analysis model wrapper."""

    def setUp(self):
        setup_training_step2(self)

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

    def test_fold_validation(self):
        """
        Check that the KFoldValidation process, utilizing the LinearDiscriminantAnalysis model,
        successfully processes the training sets and populates both the reports
        and the contingency tables via the validate method.
        """
        run_fold_validation(self)


class TestSVM(unittest.TestCase):
    """Tests for the SVM model wrapper."""

    def setUp(self):
        setup_training_step2(self)

        self.config.data["step_class_set"]["steps"]["model"] = "SVM"

    def test_base_model(self):
        """
        Ensure the base model attribute of KFoldValidation is an SVM
        instance, as defined by the configuration.
        """
        ds = KFoldValidation(self.config)
        self.assertIsInstance(ds.base_model, SupportVectorMachine)

    def test_fold_validation(self):
        """
        Check that the KFoldValidation process, utilizing the SVM model,
        successfully processes the training sets and populates both the reports
        and the contingency tables via the validate method.
        """
        run_fold_validation(self)


class TestDecisionTree(unittest.TestCase):
    """Tests for the Decision Tree model wrapper."""

    def setUp(self):
        setup_training_step2(self)

        self.config.data["step_class_set"]["steps"]["model"] = "DecisionTree"

    def test_base_model(self):
        """
        Ensure the base model attribute of KFoldValidation is a DecisionTree
        instance, as defined by the configuration.
        """
        ds = KFoldValidation(self.config)
        self.assertIsInstance(ds.base_model, DecisionTree)

    def test_fold_validation(self):
        """
        Check that the KFoldValidation process, utilizing the DecisionTree model,
        successfully processes the training sets and populates both the reports
        and the contingency tables via the validate method.
        """
        run_fold_validation(self)


class TestRandomForest(unittest.TestCase):
    """Tests for the Random Forest model wrapper."""

    def setUp(self):
        setup_training_step2(self)

        self.config.data["step_class_set"]["steps"]["model"] = "RandomForest"

    def test_base_model(self):
        """
        Ensure the base model attribute of KFoldValidation is a RandomForest
        instance, as defined by the configuration.
        """
        ds = KFoldValidation(self.config)
        self.assertIsInstance(ds.base_model, RandomForest)

    def test_fold_validation(self):
        """
        Check that the KFoldValidation process, utilizing the RandomForest model,
        successfully processes the training sets and populates both the reports
        and the contingency tables via the validate method.
        """
        run_fold_validation(self)


class TestKNN(unittest.TestCase):
    """Tests for the K-Nearest Neighbors model wrapper."""

    def setUp(self):
        setup_training_step2(self)

        self.config.data["step_class_set"]["steps"]["model"] = "KNearestNeighbors"

    def test_base_model(self):
        """
        Ensure the base model attribute of KFoldValidation is a KNearestNeighbors
        instance, as defined by the configuration.
        """
        ds = KFoldValidation(self.config)
        self.assertIsInstance(ds.base_model, KNearestNeighbors)

    def test_fold_validation(self):
        """
        Check that the KFoldValidation process, utilizing the KNearestNeighbors model,
        successfully processes the training sets and populates both the reports
        and the contingency tables via the validate method.
        """
        run_fold_validation(self)


class TestGaussianNaiveBayes(unittest.TestCase):
    """Tests for the Gaussian Naive Bayes model wrapper."""

    def setUp(self):
        setup_training_step2(self)

        self.config.data["step_class_set"]["steps"]["model"] = "GaussianNaiveBayes"

    def test_base_model(self):
        """
        Ensure the base model attribute of KFoldValidation is a GaussianNaiveBayes
        instance, as defined by the configuration.
        """
        ds = KFoldValidation(self.config)
        self.assertIsInstance(ds.base_model, GaussianNaiveBayes)

    def test_fold_validation(self):
        """
        Check that the KFoldValidation process, utilizing the GaussianNaiveBayes model,
        successfully processes the training sets and populates both the reports
        and the contingency tables via the validate method.
        """
        run_fold_validation(self)


class TestMLP(unittest.TestCase):
    """Tests for the Multi-layer Perceptron model wrapper."""

    def setUp(self):
        setup_training_step2(self)

        self.config.data["step_class_set"]["steps"]["model"] = "MLP"

    def test_base_model(self):
        """
        Ensure the base model attribute of KFoldValidation is a MLP
        instance, as defined by the configuration.
        """
        ds = KFoldValidation(self.config)
        self.assertIsInstance(ds.base_model, MultilayerPerceptron)

    def test_fold_validation(self):
        """
        Check that the KFoldValidation process, utilizing the MLP model,
        successfully processes the training sets and populates both the reports
        and the contingency tables via the validate method.
        """
        run_fold_validation(self)
