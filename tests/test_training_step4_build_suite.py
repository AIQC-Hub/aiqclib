"""
This module contains unit tests for the BuildModelSuite class, which is responsible
for building, testing, and saving multiple machine learning models concurrently
within the aiqclib training pipeline.
"""

import os
import unittest
from pathlib import Path

import matplotlib
import polars as pl

# Use non-interactive backend to prevent plots from trying to open windows during tests
matplotlib.use("Agg")

from aiqclib.common.config.training_config import TrainingConfig
from aiqclib.common.loader.training_loader import load_step1_input_training_set
from aiqclib.train.models.model_suite import ModelSuite
from aiqclib.train.step4_build_model.build_model_suite import BuildModelSuite


def setup_training_step4(test_obj):
    """
    Prepare a test training configuration and load input data for subsequent tests.
    Forces the configuration to use the ModelSuite and limits it to two methods.
    """
    test_obj.config_file_path = (
        Path(__file__).resolve().parent / "data" / "config" / "test_training_001.yaml"
    )
    test_obj.config = TrainingConfig(str(test_obj.config_file_path))
    test_obj.config.select("NRT_BO_001")

    # Force the configuration to use BuildModelSuite
    test_obj.config.data["step_class_set"]["steps"]["build"] = "BuildModelSuite"
    # Force configuration to use ModelSuite instead of a single model
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


class TestBuildModelSuite(unittest.TestCase):
    """
    A suite of tests ensuring that building, testing, and saving multiple models
    via BuildModelSuite follows the expected configuration and aggregation flows.
    """

    def setUp(self):
        """
        Prepare a test training configuration and load input data for subsequent tests.
        """
        setup_training_step4(self)

    def test_step_name(self):
        """Check that the BuildModelSuite step name is correctly assigned."""
        ds = BuildModelSuite(self.config)
        self.assertEqual(ds.step_name, "build")

    def test_multi_flag_check(self):
        """
        Verify that instantiating BuildModelSuite with a standard model
        (multi=False) raises a ValueError.
        """
        self.config.data["step_class_set"]["steps"]["model"] = "XGBoost"
        with self.assertRaisesRegex(ValueError, "multi=True"):
            _ = BuildModelSuite(self.config)

    def test_shap_flag(self):
        ds = BuildModelSuite(self.config)
        model = ds.base_model
        self.assertFalse(model.enable_shap)
        for method_obj in model.method_objs.values():
            self.assertFalse(method_obj.enable_shap)

        self.config.data["step_param_set"]["steps"]["model"]["calculate_shap"] = True
        ds = BuildModelSuite(self.config)
        model = ds.base_model
        self.assertTrue(model.enable_shap)
        for method_obj in model.method_objs.values():
            self.assertTrue(method_obj.enable_shap)

        self.config.data["step_param_set"]["steps"]["model"]["calculate_shap"] = False
        ds = BuildModelSuite(self.config)
        model = ds.base_model
        self.assertFalse(model.enable_shap)
        for method_obj in model.method_objs.values():
            self.assertFalse(method_obj.enable_shap)

    def test_output_file_names(self):
        """
        Verify that default output file names correctly reflect composite keys
        for models, and aggregated target keys for reports/predictions.
        """
        ds = BuildModelSuite(self.config)

        # Model files should be uniquely named using the composite key
        self.assertEqual(
            "/path/to/model_1/nrt_bo_001/model_folder_1/model_xgb_temp.joblib",
            str(ds.model_file_names["xgb_temp"]),
        )
        self.assertEqual(
            "/path/to/model_1/nrt_bo_001/model_folder_1/model_dt_psal.joblib",
            str(ds.model_file_names["dt_psal"]),
        )

        # Aggregated result files should NOT contain the method name, only the target
        self.assertEqual(
            "/path/to/build_1/nrt_bo_001/build_folder_1/test_report_temp.tsv",
            str(ds.output_file_names["report"]["temp"]),
        )
        self.assertEqual(
            "/path/to/build_1/nrt_bo_001/build_folder_1/test_contingency_tables_psal.parquet",
            str(ds.output_file_names["contingency_table"]["psal"]),
        )
        self.assertEqual(
            "/path/to/build_1/nrt_bo_001/build_folder_1/test_shap_values_psal.parquet",
            str(ds.output_file_names["shap_value"]["psal"]),
        )
        self.assertEqual(
            "/path/to/build_1/nrt_bo_001/build_folder_1/test_metric_plots_pres.svg",
            str(ds.output_file_names["metric_plot"]["pres"]),
        )

    def test_base_model(self):
        """Ensure that the configured base model is a ModelSuite instance."""
        ds = BuildModelSuite(self.config)
        self.assertIsInstance(ds.base_model, ModelSuite)

    def test_build_final_model_targets(self):
        """Confirm that building models populates 'final_models' with composite keys."""
        ds = BuildModelSuite(
            self.config,
            training_sets=self.ds_input.training_sets,
            test_sets=self.ds_input.test_sets,
        )
        ds.build_final_model_targets()

        # Both methods should exist for all targets
        self.assertIn("xgb_temp", ds.final_models)
        self.assertIn("dt_temp", ds.final_models)
        self.assertIsNot(ds.final_models["xgb_temp"], ds.final_models["dt_temp"])

        # Verify that internal data was joined correctly
        self.assertEqual(
            ds.final_models["xgb_temp"].training_set.height, 116 + 12
        )  # 116 train + 12 test

    def test_build_targets(self):
        """Confirm that building models populates 'models' with composite keys."""
        ds = BuildModelSuite(
            self.config,
            training_sets=self.ds_input.training_sets,
            test_sets=self.ds_input.test_sets,
        )
        ds.build_targets()

        # Both methods should exist for all targets
        self.assertIn("xgb_temp", ds.models)
        self.assertIn("dt_temp", ds.models)
        self.assertIsNot(ds.models["xgb_temp"], ds.models["dt_temp"])

        # Verify that internal data was joined correctly
        self.assertEqual(ds.models["xgb_temp"].training_set.height, 116)

    def test_build_without_data(self):
        """Ensure that calling build_targets() without data raises ValueError."""
        ds = BuildModelSuite(self.config, training_sets=None, test_sets=None)
        with self.assertRaises(ValueError):
            ds.build_targets()

    def test_build_final_model_without_test_data(self):
        """Ensure that calling build_final_model_targets() without data raises ValueError."""
        ds = BuildModelSuite(
            self.config, training_sets=self.ds_input.training_sets, test_sets=None
        )
        with self.assertRaises(ValueError):
            ds.build_final_model_targets()

    def test_build_final_model_without_training_data(self):
        """Ensure that calling build_final_model_targets() without data raises ValueError."""
        ds = BuildModelSuite(self.config, training_sets=None, test_sets=None)
        with self.assertRaises(ValueError):
            ds.build_final_model_targets()

    def test_test_targets(self):
        """
        Check that testing sets populates aggregated result columns including
        the newly introduced 'method' column.
        """
        ds = BuildModelSuite(
            self.config,
            training_sets=self.ds_input.training_sets,
            test_sets=self.ds_input.test_sets,
        )
        ds.build_targets()
        ds.test_targets()

        # Check aggregated Predictions
        self.assertIsInstance(ds.predictions["temp"], pl.DataFrame)
        self.assertIn("method", ds.predictions["temp"].columns)
        # Temp has 12 test rows. 2 models = 24 rows total.
        self.assertEqual(ds.predictions["temp"].shape[0], 24)

        # Check aggregated Contingency Tables
        self.assertIsInstance(ds.contingency_tables["psal"], pl.DataFrame)
        self.assertIn("method", ds.contingency_tables["psal"].columns)
        # Psal has 14 test rows. 2 models = 28 rows total.
        self.assertEqual(ds.contingency_tables["psal"].height, 28)

        # Check aggregated Reports
        self.assertIsInstance(ds.reports["pres"], pl.DataFrame)
        self.assertIn("method", ds.reports["pres"].columns)

    def test_test_without_model(self):
        """Ensure that calling test_targets() without first building models raises a ValueError."""
        ds = BuildModelSuite(
            self.config,
            training_sets=self.ds_input.training_sets,
            test_sets=self.ds_input.test_sets,
        )
        with self.assertRaises(ValueError):
            ds.test_targets()

    def test_write_aggregated_results(self):
        """
        Check that aggregated reports, contingency tables, and predictions are written.
        """
        self.config.data["step_param_set"]["steps"]["model"]["calculate_shap"] = True
        ds = BuildModelSuite(
            self.config,
            training_sets=self.ds_input.training_sets,
            test_sets=self.ds_input.test_sets,
        )
        data_path = Path(__file__).resolve().parent / "data" / "training"

        # Override output file names to a local temporary location
        ds.output_file_names["report"]["temp"] = str(
            data_path / "temp_test_report_temp.tsv"
        )
        ds.output_file_names["contingency_table"]["temp"] = str(
            data_path / "temp_test_contingency_temp.parquet"
        )
        ds.output_file_names["shap_value"]["temp"] = str(
            data_path / "temp_test_shap_values_temp.parquet"
        )
        ds.output_file_names["prediction"]["temp"] = str(
            data_path / "temp_test_prediction_temp.parquet"
        )
        ds.output_file_names["metric_plot"]["temp"] = str(
            data_path / "temp_test_metric_plot_temp.svg"
        )

        ds.output_file_names["report"]["psal"] = str(
            data_path / "temp_test_report_psal.tsv"
        )
        ds.output_file_names["contingency_table"]["psal"] = str(
            data_path / "temp_test_contingency_psal.parquet"
        )
        ds.output_file_names["shap_value"]["psal"] = str(
            data_path / "temp_test_shap_values_psal.parquet"
        )
        ds.output_file_names["prediction"]["psal"] = str(
            data_path / "temp_test_prediction_psal.parquet"
        )
        ds.output_file_names["metric_plot"]["psal"] = str(
            data_path / "temp_test_metric_plot_psal.svg"
        )

        ds.output_file_names["report"]["pres"] = str(
            data_path / "temp_test_report_pres.tsv"
        )
        ds.output_file_names["contingency_table"]["pres"] = str(
            data_path / "temp_test_contingency_pres.parquet"
        )
        ds.output_file_names["shap_value"]["pres"] = str(
            data_path / "temp_test_shap_values_pres.parquet"
        )
        ds.output_file_names["prediction"]["pres"] = str(
            data_path / "temp_test_prediction_pres.parquet"
        )
        ds.output_file_names["metric_plot"]["pres"] = str(
            data_path / "temp_test_metric_plot_pres.svg"
        )

        ds.build_targets()
        ds.test_targets()

        ds.write_reports()
        ds.write_contingency_tables()
        ds.write_shap_values()
        ds.write_predictions()
        ds.create_metric_plots()

        self.assertTrue(os.path.exists(ds.output_file_names["report"]["temp"]))
        self.assertTrue(
            os.path.exists(ds.output_file_names["contingency_table"]["temp"])
        )
        self.assertTrue(os.path.exists(ds.output_file_names["shap_value"]["temp"]))
        self.assertTrue(os.path.exists(ds.output_file_names["prediction"]["temp"]))
        self.assertTrue(os.path.exists(ds.output_file_names["metric_plot"]["temp"]))

        self.assertTrue(os.path.exists(ds.output_file_names["report"]["psal"]))
        self.assertTrue(
            os.path.exists(ds.output_file_names["contingency_table"]["psal"])
        )
        self.assertTrue(os.path.exists(ds.output_file_names["shap_value"]["psal"]))
        self.assertTrue(os.path.exists(ds.output_file_names["prediction"]["psal"]))
        self.assertTrue(os.path.exists(ds.output_file_names["metric_plot"]["psal"]))

        self.assertTrue(os.path.exists(ds.output_file_names["report"]["pres"]))
        self.assertTrue(
            os.path.exists(ds.output_file_names["contingency_table"]["pres"])
        )
        self.assertTrue(os.path.exists(ds.output_file_names["shap_value"]["pres"]))
        self.assertTrue(os.path.exists(ds.output_file_names["prediction"]["pres"]))
        self.assertTrue(os.path.exists(ds.output_file_names["metric_plot"]["pres"]))

        # Cleanup
        os.remove(ds.output_file_names["report"]["temp"])
        os.remove(ds.output_file_names["contingency_table"]["temp"])
        os.remove(ds.output_file_names["shap_value"]["temp"])
        os.remove(ds.output_file_names["prediction"]["temp"])
        os.remove(ds.output_file_names["metric_plot"]["temp"])

        os.remove(ds.output_file_names["report"]["psal"])
        os.remove(ds.output_file_names["contingency_table"]["psal"])
        os.remove(ds.output_file_names["shap_value"]["psal"])
        os.remove(ds.output_file_names["prediction"]["psal"])
        os.remove(ds.output_file_names["metric_plot"]["psal"])

        os.remove(ds.output_file_names["report"]["pres"])
        os.remove(ds.output_file_names["contingency_table"]["pres"])
        os.remove(ds.output_file_names["shap_value"]["pres"])
        os.remove(ds.output_file_names["prediction"]["pres"])
        os.remove(ds.output_file_names["metric_plot"]["pres"])

    def test_write_models(self):
        """
        Check that individual trained models are serialized to files correctly.
        """
        ds = BuildModelSuite(
            self.config,
            training_sets=self.ds_input.training_sets,
            test_sets=self.ds_input.test_sets,
        )
        data_path = Path(__file__).resolve().parent / "data" / "training"

        # Override paths for composite keys
        ds.model_file_names["xgb_temp"] = str(data_path / "temp_model_xgb_temp.joblib")
        ds.model_file_names["xgb_psal"] = str(data_path / "temp_model_xgb_psal.joblib")
        ds.model_file_names["xgb_pres"] = str(data_path / "temp_model_xgb_pres.joblib")
        ds.model_file_names["dt_temp"] = str(data_path / "temp_model_dt_temp.joblib")
        ds.model_file_names["dt_psal"] = str(data_path / "temp_model_dt_psal.joblib")
        ds.model_file_names["dt_pres"] = str(data_path / "temp_model_dt_pres.joblib")

        ds.build_final_model_targets()
        ds.write_models()

        self.assertTrue(os.path.exists(ds.model_file_names["xgb_temp"]))
        self.assertTrue(os.path.exists(ds.model_file_names["xgb_psal"]))
        self.assertTrue(os.path.exists(ds.model_file_names["xgb_pres"]))
        self.assertTrue(os.path.exists(ds.model_file_names["dt_temp"]))
        self.assertTrue(os.path.exists(ds.model_file_names["dt_psal"]))
        self.assertTrue(os.path.exists(ds.model_file_names["dt_pres"]))

        os.remove(ds.model_file_names["xgb_temp"])
        os.remove(ds.model_file_names["xgb_psal"])
        os.remove(ds.model_file_names["xgb_pres"])
        os.remove(ds.model_file_names["dt_temp"])
        os.remove(ds.model_file_names["dt_psal"])
        os.remove(ds.model_file_names["dt_pres"])

    def test_empty_write_calls(self):
        """
        Ensure ValueErrors are raised if write methods are called empty datasets.
        """
        ds = BuildModelSuite(
            self.config,
            training_sets=self.ds_input.training_sets,
            test_sets=self.ds_input.test_sets,
        )

        with self.assertRaises(ValueError):
            ds.write_reports()
        with self.assertRaises(ValueError):
            ds.write_contingency_tables()
        with self.assertRaises(ValueError):
            ds.create_metric_plots()
        with self.assertRaises(ValueError):
            ds.write_predictions()
        with self.assertRaises(ValueError):
            ds.write_models()

    def test_read_models_no_file(self):
        """Check that FileNotFoundError is raised if model files are missing during reading."""
        ds = BuildModelSuite(
            self.config, training_sets=None, test_sets=self.ds_input.test_sets
        )
        data_path = Path(__file__).resolve().parent / "data" / "training"
        ds.model_file_names["xgb_temp"] = str(data_path / "non_existent_model.joblib")

        with self.assertRaises(FileNotFoundError):
            ds.read_models()
