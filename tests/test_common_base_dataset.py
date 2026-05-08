"""
Unit tests for the DataSetBase class in aiqclib.common.base.dataset_base
This module verifies the correct functionality of DataSetBase's methods.
"""

import unittest
from pathlib import Path

from aiqclib.common.base.config_base import ConfigBase
from aiqclib.common.base.dataset_base import DataSetBase
from aiqclib.common.config.dataset_config import DataSetConfig


class DataSetWithExpectedName(DataSetBase):
    """
    DataSetWithExpectedName is used to test methods and variables in DataSetBase
    """

    expected_class_name: str = "InputDataSetA"

    def __init__(self, step_name: str, config: ConfigBase) -> None:
        super().__init__(step_name, config)


class TestDatasetBaseMethods(unittest.TestCase):
    """
    A suite of tests that verify the correctness of methods
    within the DataSetBase.
    """

    def setUp(self):
        """
        Set up a reference to the test configuration file (test_dataset_001.yaml)
        to be used by all subsequent tests in this class.
        """
        self.config_file_path = (
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_dataset_001.yaml"
        )
        self.config = DataSetConfig(str(self.config_file_path))
        self.config.select("NRT_BO_001")

    def test_common_base_path(self):
        """
        Verifies that direct instantiation of DataSetBase (an abstract base class) raises NotImplementedError.
        """
        with self.assertRaises(NotImplementedError):
            _ = DataSetBase("input", self.config)

    def test_step_name(self):
        """
        Ensures that creating a DataSet instance with an unmatched step name raises a ValueError.
        """
        with self.assertRaises(ValueError):
            _ = DataSetWithExpectedName("select", self.config)

    def test_represented_str(self):
        """
        Ensures that the DataSet instance returns a correct string representation via its __str__ method.
        """

        ds = DataSetWithExpectedName("input", self.config)
        self.assertEqual(str(ds), "DataSetBase(step=input, class=InputDataSetA)")
