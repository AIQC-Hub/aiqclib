"""
Unit tests for the ModelSuite class.

This module verifies that ModelSuite correctly parses the configuration,
loads the specified machine learning methods using the actual loader, and
ensures that specific configuration parameters are effectively applied to
the correct underlying model objects.
"""

import unittest
from pathlib import Path

from aiqclib.common.config.training_config import TrainingConfig
from aiqclib.train.models.model_suite import ModelSuite


class TestModelSuite(unittest.TestCase):
    """
    A suite of tests to verify the initialization, method loading,
    and parameter assignment behavior of the ModelSuite class.
    """

    def setUp(self):
        """
        Set up the configuration object using a test YAML file.
        """
        self.config_file_path = (
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_training_001.yaml"
        )
        self.config = TrainingConfig(str(self.config_file_path))
        self.config.select("NRT_BO_001")

        # Ensure the config passes the expected_class_name validation in ModelBase
        self.config.data["step_class_set"]["steps"]["model"] = "ModelSuite"

    def test_multi_flag(self):
        ds = ModelSuite(self.config)
        self.assertTrue(ds.multi)

    def test_shap_flag(self):
        ds = ModelSuite(self.config)
        self.assertFalse(ds.enable_shap)
        for method_obj in ds.method_objs.values():
            self.assertFalse(method_obj.enable_shap)

        self.config.data["step_param_set"]["steps"]["model"]["calculate_shap"] = True
        ds = ModelSuite(self.config)
        self.assertTrue(ds.enable_shap)
        for method_obj in ds.method_objs.values():
            self.assertTrue(method_obj.enable_shap)

        self.config.data["step_param_set"]["steps"]["model"]["calculate_shap"] = False
        ds = ModelSuite(self.config)
        self.assertFalse(ds.enable_shap)
        for method_obj in ds.method_objs.values():
            self.assertFalse(method_obj.enable_shap)

    def test_init_default_methods(self):
        """
        Verify that ModelSuite loads all default methods when the config
        does not explicitly specify a 'methods' list.
        """
        # Clear any specific model parameters or methods from the base config
        self.config.data["step_param_set"]["steps"]["model"] = {}

        suite = ModelSuite(self.config)

        # Assert basic properties
        self.assertEqual(suite.expected_class_name, "ModelSuite")
        self.assertEqual(suite.short_name, "MS")

        # Verify method_objs contains exactly the 9 default methods
        self.assertEqual(len(suite.method_objs), 9)
        for method_name in suite.default_methods:
            self.assertIn(method_name, suite.method_objs)

            # Check that an actual object was loaded (not None)
            self.assertIsNotNone(suite.method_objs[method_name])

    def test_init_custom_methods_with_params(self):
        """
        Verify that ModelSuite loads only the specified methods and successfully
        applies the nested specific parameters from the config file to the
        correct model objects.
        """
        # Simulate the YAML config snippet provided
        self.config.data["step_param_set"]["steps"]["model"] = {
            "methods": ["DT", "XGB", "RF"],
            "model_params": {
                "DT": {"class_weight": "balanced"},
                "RF": {"class_weight": "balanced"},
            },
        }

        suite = ModelSuite(self.config)

        # Verify method_objs contains exactly the 3 custom methods specified
        self.assertEqual(len(suite.method_objs), 3)
        self.assertIn("DT", suite.method_objs)
        self.assertIn("XGB", suite.method_objs)
        self.assertIn("RF", suite.method_objs)

        # Verify that specific parameters took effect for DT
        dt_model = suite.method_objs["DT"]
        self.assertIn("class_weight", dt_model.model_params)
        self.assertEqual(dt_model.model_params["class_weight"], "balanced")

        # Verify that specific parameters took effect for RF
        rf_model = suite.method_objs["RF"]
        self.assertIn("class_weight", rf_model.model_params)
        self.assertEqual(rf_model.model_params["class_weight"], "balanced")

        # Verify XGB didn't mistakenly receive the class_weight param,
        # and that it retained its own default values (e.g., n_jobs)
        xgb_model = suite.method_objs["XGB"]
        self.assertNotIn("class_weight", xgb_model.model_params)
        self.assertIn("n_jobs", xgb_model.model_params)

    def test_long_methods(self):
        """
        Verify that ModelSuite loads only the specified methods and successfully
        applies the nested specific parameters from the config file to the
        correct model objects.
        """
        # Simulate the YAML config snippet provided
        self.config.data["step_param_set"]["steps"]["model"] = {
            "methods": ["DecisionTree", "XGBoost", "RandomForest"],
            "model_params": {
                "DecisionTree": {"class_weight": "balanced"},
                "RandomForest": {"class_weight": "balanced"},
            },
        }

        suite = ModelSuite(self.config)

        # Verify method_objs contains exactly the 3 custom methods specified
        self.assertEqual(len(suite.method_objs), 3)
        self.assertIn("DecisionTree", suite.method_objs)
        self.assertIn("XGBoost", suite.method_objs)
        self.assertIn("RandomForest", suite.method_objs)

        # Verify that specific parameters took effect for DT
        dt_model = suite.method_objs["DecisionTree"]
        self.assertIn("class_weight", dt_model.model_params)
        self.assertEqual(dt_model.model_params["class_weight"], "balanced")

        # Verify that specific parameters took effect for RF
        rf_model = suite.method_objs["RandomForest"]
        self.assertIn("class_weight", rf_model.model_params)
        self.assertEqual(rf_model.model_params["class_weight"], "balanced")

        # Verify XGB didn't mistakenly receive the class_weight param,
        # and that it retained its own default values (e.g., n_jobs)
        xgb_model = suite.method_objs["XGBoost"]
        self.assertNotIn("class_weight", xgb_model.model_params)
        self.assertIn("n_jobs", xgb_model.model_params)

    def test_override_existing_defaults(self):
        """
        Verify that values in the config file successfully override the
        default parameters of the loaded ML objects.
        """
        self.config.data["step_param_set"]["steps"]["model"] = {
            "methods": ["XGB", "KNN"],
            "model_params": {
                "XGB": {"n_estimators": 500, "max_depth": 3},
                "KNN": {"n_neighbors": 15},
            },
        }

        suite = ModelSuite(self.config)

        # Check XGB defaults were overridden
        xgb_model = suite.method_objs["XGB"]
        self.assertEqual(xgb_model.model_params["n_estimators"], 500)
        self.assertEqual(xgb_model.model_params["max_depth"], 3)

        # Check KNN defaults were overridden
        knn_model = suite.method_objs["KNN"]
        self.assertEqual(knn_model.model_params["n_neighbors"], 15)
