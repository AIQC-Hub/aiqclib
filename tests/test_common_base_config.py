"""Unit tests for the ConfigBase class in aiqclib.common.base.config_base.

This module verifies the correct functionality of ConfigBase's methods,
including initialization, validation, and data retrieval, as well as the
correct loading of configuration templates.
"""

import unittest
import pytest
from pathlib import Path

from aiqclib.common.base.config_base import ConfigBase
from aiqclib.common.config.classify_config import ClassificationConfig
from aiqclib.common.config.dataset_config import DataSetConfig
from aiqclib.common.config.training_config import TrainingConfig


class ConfigBaseWithExpectedName(ConfigBase):
    """A helper class for testing ConfigBase.

    This class extends ConfigBase and defines the `expected_class_name`
    attribute, allowing it to be instantiated and used in tests.
    """

    expected_class_name: str = "ConfigBaseWithExpectedName"

    def __init__(self, section_name: str, config_file: str) -> None:
        """Initialize a new instance of ConfigBaseWithExpectedName.

        This constructor calls the parent `ConfigBase` constructor with the
        provided section name and configuration file path.
        """
        super().__init__(section_name, config_file)


class TestDatasetBaseMethods(unittest.TestCase):
    """A suite of tests that verify the correctness of methods
    within the ConfigBase class.
    """

    def setUp(self):
        """Set up the path to the test configuration file.

        This method is called before each test function to ensure the
        `self.config_file_path` is correctly set.
        """
        self.config_file_path = (
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_dataset_001.yaml"
        )

    def test_common_base_path(self):
        """Verify that instantiating ConfigBase directly raises NotImplementedError.

        This test checks that `ConfigBase` cannot be instantiated without a
        subclass defining `expected_class_name`.
        """
        with self.assertRaises(NotImplementedError):
            _ = ConfigBase("data_sets", self.config_file_path)

    def test_section_name(self):
        """Verify that an unsupported section name raises a ValueError.

        This test checks that initializing `ConfigBase` with a section name
        not in its supported list raises an error.
        """
        with self.assertRaises(ValueError):
            _ = ConfigBaseWithExpectedName(
                "invalid_section_name", self.config_file_path
            )

    def test_represented_str(self):
        """Verify that the instance returns a correct string representation.

        This test checks the output of the `__repr__` method of the `ConfigBase`
        instance.
        """
        ds = ConfigBaseWithExpectedName("data_sets", self.config_file_path)
        self.assertEqual(str(ds), "ConfigBase(section_name=data_sets)")

    def test_validation_error_with_select(self):
        """Verify that invalid YAML content raises a ValueError during selection.

        This test intentionally corrupts `full_config` to simulate an invalid
        YAML structure and asserts that `select` method correctly raises a
        `ValueError` due to schema validation failure.
        """
        ds = ConfigBaseWithExpectedName("data_sets", self.config_file_path)
        ds.full_config = ""
        with self.assertRaises(ValueError):
            ds.select("NRT_BO_001")

    def test_no_base_name(self):
        """Verify that `get_base_path` raises an error if the base path is None.

        This test sets the 'common' base_path to None in the loaded configuration
        and asserts that calling `get_base_path` raises a `ValueError`.
        """
        ds = ConfigBaseWithExpectedName("data_sets", self.config_file_path)
        ds.select("NRT_BO_001")
        ds.data["path_info"]["common"]["base_path"] = None
        with self.assertRaises(ValueError):
            ds.get_base_path("invalid_step_name")


class TestConfigTemplates:
    """Tests for loading configuration from built-in YAML templates."""

    @pytest.mark.parametrize(
        "template",
        [
            (DataSetConfig, "template:data_sets", "dataset_0001"),
            (DataSetConfig, "template:data_sets_full", "dataset_0001"),
            (TrainingConfig, "template:training_sets", "training_0001"),
            (
                ClassificationConfig,
                "template:classification_sets_full",
                "classification_0001",
            ),
            (
                ClassificationConfig,
                "template:classification_sets",
                "classification_0001",
            ),
        ],
    )
    def test_read_template(self, template):
        """Verify that DataSetConfig can load and select from the 'data_sets' template.

        This test ensures that the configuration is loaded successfully and that
        a specific dataset can be selected from the template.
        """
        conf = template[0](template[1])
        assert conf.full_config is not None

        assert conf.data is None
        conf.select(template[2])
        assert conf.data is not None
