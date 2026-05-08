"""
Unit tests for verifying the correct loading and initialization of model classes
at various processing steps, using common loader functions.

This module contains tests for the `load_model_class` function and
initial state validation of loaded model instances.
"""

import unittest
from pathlib import Path

from aiqclib.common.config.training_config import TrainingConfig
from aiqclib.common.loader.model_loader import load_model_class
from aiqclib.train.models.xgboost import XGBoost
from aiqclib.train.models.model_suite import ModelSuite


class TestModelClassLoader(unittest.TestCase):
    """
    Tests related to loading the Model class.
    """

    def setUp(self):
        """
        Defines the path to the test configuration file and selects a dataset
        prior to each test execution.
        """
        self.config_file_path = (
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_training_001.yaml"
        )
        self.config = TrainingConfig(str(self.config_file_path))
        self.config.select("NRT_BO_001")

    def test_load_model_valid_config(self):
        """
        Tests that load_model_class successfully returns an XGBoost instance
        when provided with a valid configuration.
        """
        ds = load_model_class(self.config)
        self.assertIsInstance(ds, XGBoost)

        self.config.data["step_class_set"]["steps"]["model"] = "XGB"
        ds = load_model_class(self.config)
        self.assertIsInstance(ds, XGBoost)

    def test_load_model_suite(self):
        """
        Tests that load_model_class successfully returns an ModelSuite instance
        when provided with a valid configuration.
        """
        self.config.data["step_class_set"]["steps"]["model"] = "MS"
        ds = load_model_class(self.config)
        self.assertIsInstance(ds, ModelSuite)

        self.config.data["step_class_set"]["steps"]["model"] = "ModelSuite"
        ds = load_model_class(self.config)
        self.assertIsInstance(ds, ModelSuite)

    def test_load_model_invalid_config(self):
        """
        Verifies that load_model_class raises a ValueError when an
        invalid model name is specified in the configuration.
        """
        self.config.data["step_class_set"]["steps"]["model"] = "invalid_model_name"
        with self.assertRaises(ValueError):
            _ = load_model_class(self.config)

    def test_build_model_empty_training_set(self):
        """
        Ensures that the model's build method raises a ValueError
        if the training set has not been provided.
        """
        ds = load_model_class(self.config)
        with self.assertRaises(ValueError):
            ds.build()

    def test_predict_model_empty_test_set(self):
        """
        Ensures that the model's predict method raises a ValueError
        if the test set has not been provided.
        """
        ds = load_model_class(self.config)
        with self.assertRaises(ValueError):
            ds.predict()

    def test_create_report_empty_test_set(self):
        """
        Ensures that the model's create_report method raises a ValueError
        if the test set has not been provided.
        """
        ds = load_model_class(self.config)
        with self.assertRaises(ValueError):
            ds.create_report()

    def test_create_report_empty_predictions(self):
        """
        Ensures that the model's create_report method raises a ValueError
        if predictions have not been generated or set.
        """
        ds = load_model_class(self.config)
        ds.test_set = {}  # Set test_set to an empty dict to bypass `test_set not set` check
        with self.assertRaises(ValueError):
            ds.create_report()
