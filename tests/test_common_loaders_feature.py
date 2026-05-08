"""
Unit tests for verifying the correct loading and initialization of feature classes
at various processing steps, using common loader functions.
"""

import unittest
from pathlib import Path

from aiqclib.common.config.dataset_config import DataSetConfig
from aiqclib.common.loader.feature_loader import load_feature_class
from aiqclib.prepare.features.location import LocationFeat


class TestFeatureClassLoader(unittest.TestCase):
    """
    Tests related to loading the Feature class.
    """

    def setUp(self):
        """
        Define the path to the test config file and select a dataset
        prior to each test.
        """
        self.config_file_path = (
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_dataset_001.yaml"
        )
        self.config = DataSetConfig(str(self.config_file_path))
        self.config.select("NRT_BO_001")

    def test_load_model_valid_config(self):
        """
        Test that `load_feature_class` successfully loads and returns an instance
        of `LocationFeat` when provided with valid configuration parameters.
        """
        ds = load_feature_class(
            "temp", self.config.data["feature_param_set"]["params"][0]
        )
        self.assertIsInstance(ds, LocationFeat)

    def test_load_model_invalid_config(self):
        """
        Test that `load_feature_class` raises a `ValueError` when an
        invalid feature name is provided in the configuration.
        """
        self.config.data["feature_param_set"]["params"][0]["feature"] = (
            "invalid_feature_name"
        )
        with self.assertRaises(ValueError):
            _ = load_feature_class(
                "temp", self.config.data["feature_param_set"]["params"][0]
            )
