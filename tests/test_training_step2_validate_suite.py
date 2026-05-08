"""
This module contains unit tests for the KFoldValidationSuite class, ensuring its
correct integration with training configurations, data loading, the ModelSuite,
and report generation. It verifies that the validation process
behaves as expected across various scenarios and multiple ML algorithms.
"""

import os
import unittest
from pathlib import Path

import polars as pl

from aiqclib.common.config.training_config import TrainingConfig
from aiqclib.common.loader.training_loader import load_step1_input_training_set
from aiqclib.train.models.model_suite import ModelSuite
from aiqclib.train.step2_validate_model.kfold_validation_suite import (
    KFoldValidationSuite,
)


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

    # Force the configuration to use KFoldValidationSuite
    test_obj.config.data["step_class_set"]["steps"]["validate"] = "KFoldValidationSuite"
    # Force the configuration to use ModelSuite instead of a single model
    test_obj.config.data["step_class_set"]["steps"]["model"] = "ModelSuite"
    # To keep tests fast, we only test two models instead of all 9 defaults
    test_obj.config.data["step_param_set"]["steps"]["model"] = {
        "methods": ["XGB", "DT"]
    }

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


class TestKFoldValidationSuite(unittest.TestCase):
    """
    A suite of tests ensuring that KFoldValidationSuite correctly captures
    configurations, splits training data, iterates over the ModelSuite methods,
    and writes validation results with dynamic file names.
    """

    def setUp(self):
        """
        Prepare the test environment by loading a training configuration
        and input training data.
        """
        setup_training_step2(self)

    def test_step_name(self):
        """
        Check that the step name is correctly identified as 'validate'.
        """
        ds = KFoldValidationSuite(self.config)
        self.assertEqual(ds.step_name, "validate")

    def test_multi_flag_check(self):
        """
        Verify that instantiating KFoldValidationSuite with a standard model
        (multi=False) raises a ValueError.
        """
        # Revert config to standard XGBoost
        self.config.data["step_class_set"]["steps"]["model"] = "XGBoost"

        with self.assertRaisesRegex(ValueError, "multi=True"):
            _ = KFoldValidationSuite(self.config)

    def test_shap_flag(self):
        ds = KFoldValidationSuite(self.config)
        model = ds.base_model
        self.assertFalse(model.enable_shap)
        for method_obj in model.method_objs.values():
            self.assertFalse(method_obj.enable_shap)

        self.config.data["step_param_set"]["steps"]["model"]["calculate_shap"] = True
        ds = KFoldValidationSuite(self.config)
        model = ds.base_model
        self.assertFalse(model.enable_shap)
        for method_obj in model.method_objs.values():
            self.assertFalse(method_obj.enable_shap)

        self.config.data["step_param_set"]["steps"]["model"]["calculate_shap"] = False
        ds = KFoldValidationSuite(self.config)
        model = ds.base_model
        self.assertFalse(model.enable_shap)
        for method_obj in model.method_objs.values():
            self.assertFalse(method_obj.enable_shap)

    def test_output_file_names_init(self):
        """
        Verify that the default output file names initially retain the {method}
        placeholder before process_targets is run.
        """
        ds = KFoldValidationSuite(self.config)

        # Check report file names still contain {method}
        self.assertEqual(
            "/path/to/validate_1/nrt_bo_001/validate_folder_1/validation_report_{method}_temp.tsv",
            str(ds.output_file_names["report"]["temp"]),
        )
        self.assertEqual(
            "/path/to/validate_1/nrt_bo_001/validate_folder_1/contingency_tables_{method}_temp.parquet",
            str(ds.output_file_names["contingency_table"]["temp"]),
        )
        self.assertEqual(
            "/path/to/validate_1/nrt_bo_001/validate_folder_1/metric_plots_{method}_temp.svg",
            str(ds.output_file_names["metric_plot"]["temp"]),
        )

    def test_base_model(self):
        """
        Ensure the base model attribute of KFoldValidationSuite is a ModelSuite
        instance.
        """
        ds = KFoldValidationSuite(self.config)
        self.assertIsInstance(ds.base_model, ModelSuite)

    def test_fold_validation(self):
        """
        Check that the KFoldValidationSuite process successfully processes the
        training sets across multiple methods and populates the composite keys.
        """
        ds = KFoldValidationSuite(
            self.config, training_sets=self.ds_input.training_sets
        )
        ds.process_targets()

        # Check Reports - ensure composite keys exist
        self.assertIn("xgb_temp", ds.reports)
        self.assertIn("dt_temp", ds.reports)

        # Validate shapes for XGBoost Temp
        self.assertIsInstance(ds.reports["xgb_temp"], pl.DataFrame)
        self.assertEqual(ds.reports["xgb_temp"].shape[0], 18)
        self.assertEqual(ds.reports["xgb_temp"].shape[1], 8)

        # Check Contingency Tables for DT Psal
        self.assertIn("dt_psal", ds.contingency_tables)
        self.assertIsInstance(ds.contingency_tables["dt_psal"], pl.DataFrame)
        self.assertEqual(ds.contingency_tables["dt_psal"].height, 126)
        self.assertListEqual(
            ds.contingency_tables["dt_psal"].columns,
            ["k", "label", "predicted_label", "score"],
        )

        # Check that file paths were dynamically updated for the composite keys
        self.assertEqual(
            "/path/to/validate_1/nrt_bo_001/validate_folder_1/validation_report_xgb_temp.tsv",
            str(ds.output_file_names["report"]["xgb_temp"]),
        )

    def test_write_results(self):
        """
        Ensure validation reports are written to the specified output files
        with composite keys and that temporary files are cleaned up.
        """
        ds = KFoldValidationSuite(
            self.config, training_sets=self.ds_input.training_sets
        )
        ds.process_targets()

        data_path = Path(__file__).resolve().parent / "data" / "training"

        # Override specific composite key paths for testing
        ds.output_file_names["report"]["xgb_temp"] = str(
            data_path / "temp_validation_report_xgb_temp.tsv"
        )
        ds.output_file_names["report"]["xgb_psal"] = str(
            data_path / "temp_validation_report_xgb_psal.tsv"
        )
        ds.output_file_names["report"]["xgb_pres"] = str(
            data_path / "temp_validation_report_xgb_pres.tsv"
        )
        ds.output_file_names["report"]["dt_temp"] = str(
            data_path / "temp_validation_report_dt_temp.tsv"
        )
        ds.output_file_names["report"]["dt_psal"] = str(
            data_path / "temp_validation_report_dt_psal.tsv"
        )
        ds.output_file_names["report"]["dt_pres"] = str(
            data_path / "temp_validation_report_dt_pres.tsv"
        )

        ds.write_reports()

        self.assertTrue(os.path.exists(ds.output_file_names["report"]["xgb_temp"]))
        self.assertTrue(os.path.exists(ds.output_file_names["report"]["xgb_psal"]))
        self.assertTrue(os.path.exists(ds.output_file_names["report"]["xgb_pres"]))
        self.assertTrue(os.path.exists(ds.output_file_names["report"]["dt_temp"]))
        self.assertTrue(os.path.exists(ds.output_file_names["report"]["dt_psal"]))
        self.assertTrue(os.path.exists(ds.output_file_names["report"]["dt_pres"]))

        os.remove(ds.output_file_names["report"]["xgb_temp"])
        os.remove(ds.output_file_names["report"]["xgb_psal"])
        os.remove(ds.output_file_names["report"]["xgb_pres"])
        os.remove(ds.output_file_names["report"]["dt_temp"])
        os.remove(ds.output_file_names["report"]["dt_psal"])
        os.remove(ds.output_file_names["report"]["dt_pres"])

    def test_write_contingency_tables(self):
        """
        Ensure contingency tables are written to the specified output files.
        """
        ds = KFoldValidationSuite(
            self.config, training_sets=self.ds_input.training_sets
        )
        ds.process_targets()

        data_path = Path(__file__).resolve().parent / "data" / "training"

        # Override output paths for testing
        ds.output_file_names["contingency_table"]["xgb_temp"] = str(
            data_path / "temp_contingency_xgb_temp.parquet"
        )
        ds.output_file_names["contingency_table"]["xgb_psal"] = str(
            data_path / "temp_contingency_xgb_psal.parquet"
        )
        ds.output_file_names["contingency_table"]["xgb_pres"] = str(
            data_path / "temp_contingency_xgb_pres.tsv"
        )
        ds.output_file_names["contingency_table"]["dt_temp"] = str(
            data_path / "temp_contingency_dt_temp.tsv"
        )
        ds.output_file_names["contingency_table"]["dt_psal"] = str(
            data_path / "temp_contingency_dt_psal.tsv"
        )
        ds.output_file_names["contingency_table"]["dt_pres"] = str(
            data_path / "temp_contingency_dt_pres.tsv"
        )

        ds.write_contingency_tables()

        self.assertTrue(
            os.path.exists(ds.output_file_names["contingency_table"]["xgb_temp"])
        )
        self.assertTrue(
            os.path.exists(ds.output_file_names["contingency_table"]["xgb_psal"])
        )
        self.assertTrue(
            os.path.exists(ds.output_file_names["contingency_table"]["xgb_pres"])
        )
        self.assertTrue(
            os.path.exists(ds.output_file_names["contingency_table"]["dt_temp"])
        )
        self.assertTrue(
            os.path.exists(ds.output_file_names["contingency_table"]["dt_psal"])
        )
        self.assertTrue(
            os.path.exists(ds.output_file_names["contingency_table"]["dt_pres"])
        )

        os.remove(ds.output_file_names["contingency_table"]["xgb_temp"])
        os.remove(ds.output_file_names["contingency_table"]["xgb_psal"])
        os.remove(ds.output_file_names["contingency_table"]["xgb_pres"])
        os.remove(ds.output_file_names["contingency_table"]["dt_temp"])
        os.remove(ds.output_file_names["contingency_table"]["dt_psal"])
        os.remove(ds.output_file_names["contingency_table"]["dt_pres"])

    def test_create_metric_plots(self):
        """
        Ensure ROC and Precision-Recall plots written to the specified output files.
        """
        import matplotlib

        matplotlib.use("Agg")  # Prevent plots popping up during testing

        ds = KFoldValidationSuite(
            self.config, training_sets=self.ds_input.training_sets
        )
        ds.process_targets()

        data_path = Path(__file__).resolve().parent / "data" / "training"

        # Override output paths for testing
        ds.output_file_names["metric_plot"]["xgb_temp"] = str(
            data_path / "temp_metric_plots_xgb_temp.tsv"
        )
        ds.output_file_names["metric_plot"]["xgb_psal"] = str(
            data_path / "temp_metric_plots_xgb_psal.tsv"
        )
        ds.output_file_names["metric_plot"]["xgb_pres"] = str(
            data_path / "temp_metric_plots_xgb_pres.tsv"
        )
        ds.output_file_names["metric_plot"]["dt_temp"] = str(
            data_path / "temp_metric_plots_dt_temp.tsv"
        )
        ds.output_file_names["metric_plot"]["dt_psal"] = str(
            data_path / "temp_metric_plots_dt_psal.tsv"
        )
        ds.output_file_names["metric_plot"]["dt_pres"] = str(
            data_path / "temp_metric_plots_dt_pres.tsv"
        )

        ds.create_metric_plots()

        self.assertTrue(os.path.exists(ds.output_file_names["metric_plot"]["xgb_temp"]))
        self.assertTrue(os.path.exists(ds.output_file_names["metric_plot"]["xgb_psal"]))
        self.assertTrue(os.path.exists(ds.output_file_names["metric_plot"]["xgb_pres"]))
        self.assertTrue(os.path.exists(ds.output_file_names["metric_plot"]["dt_temp"]))
        self.assertTrue(os.path.exists(ds.output_file_names["metric_plot"]["dt_psal"]))
        self.assertTrue(os.path.exists(ds.output_file_names["metric_plot"]["dt_pres"]))

        os.remove(ds.output_file_names["metric_plot"]["xgb_temp"])
        os.remove(ds.output_file_names["metric_plot"]["xgb_psal"])
        os.remove(ds.output_file_names["metric_plot"]["xgb_pres"])
        os.remove(ds.output_file_names["metric_plot"]["dt_temp"])
        os.remove(ds.output_file_names["metric_plot"]["dt_psal"])
        os.remove(ds.output_file_names["metric_plot"]["dt_pres"])

    def test_write_reports_empty_reports(self):
        """
        Ensure that calling write_reports with empty reports raises a ValueError.
        """
        ds = KFoldValidationSuite(
            self.config, training_sets=self.ds_input.training_sets
        )
        with self.assertRaises(ValueError):
            ds.write_reports()

    def test_write_contingency_tables_empty(self):
        """
        Ensure that calling write_contingency_tables with empty tables raises a ValueError.
        """
        ds = KFoldValidationSuite(
            self.config, training_sets=self.ds_input.training_sets
        )
        with self.assertRaises(ValueError):
            ds.write_contingency_tables()

    def test_create_metric_plots_empty(self):
        """
        Ensure that calling create_metric_plots with empty tables raises a ValueError.
        """
        ds = KFoldValidationSuite(
            self.config, training_sets=self.ds_input.training_sets
        )
        with self.assertRaises(ValueError):
            ds.create_metric_plots()
