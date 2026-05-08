"""
Unit tests for the create_metric_plots utility function.
This module verifies that ROC and Precision-Recall plots are generated correctly
based on provided contingency tables.
"""

import os
import shutil
import tempfile
import unittest
from typing import Dict

import polars as pl
import matplotlib

# Use non-interactive backend to prevent plots from trying to open windows during tests
matplotlib.use("Agg")

from aiqclib.common.utils.metric_plots import create_metric_plots


class MockModel:
    """
    A simple mock class to simulate the structure of ValidationBase/BuildModelBase
    required by create_metric_plots.
    """

    def __init__(self) -> None:
        self.contingency_tables: Dict[str, pl.DataFrame] = {}
        self.output_file_names: Dict[str, Dict[str, str]] = {"metric_plot": {}}


class TestCreateMetricPlots(unittest.TestCase):
    """
    Test suite for the create_metric_plots function.
    """

    def setUp(self) -> None:
        """
        Create a temporary directory to store output files during testing.
        """
        self.test_dir = tempfile.mkdtemp()
        self.mock_model = MockModel()

    def tearDown(self) -> None:
        """
        Remove the temporary directory and its contents after testing.
        """
        shutil.rmtree(self.test_dir)

    def test_empty_contingency_tables(self) -> None:
        """
        Ensure that a ValueError is raised if the model has no contingency tables.
        """
        self.mock_model.contingency_tables = {}
        with self.assertRaises(ValueError):
            create_metric_plots(self.mock_model)

    def test_single_fold_plot_generation(self) -> None:
        """
        Verify that a plot file is created for a single-fold scenario (e.g., standard test set).
        """
        target_name = "temp"
        output_file = os.path.join(self.test_dir, f"plot_{target_name}.svg")

        # Setup mock data: Single fold (k=1)
        self.mock_model.output_file_names["metric_plot"][target_name] = output_file
        self.mock_model.contingency_tables[target_name] = pl.DataFrame(
            {
                "k": [1, 1, 1, 1, 1],
                "label": [0, 0, 1, 1, 0],
                "score": [0.1, 0.2, 0.8, 0.9, 0.4],
            }
        )

        create_metric_plots(self.mock_model)

        # Assert file exists and has size > 0
        self.assertTrue(os.path.exists(output_file))
        self.assertGreater(os.path.getsize(output_file), 0)

    def test_multi_fold_plot_generation(self) -> None:
        """
        Verify that a plot file is created for a multi-fold scenario (e.g., cross-validation).
        This tests the logic path for calculating mean curves and standard deviations.
        """
        target_name = "psal"
        output_file = os.path.join(self.test_dir, f"plot_{target_name}.svg")

        # Setup mock data: Two folds (k=1, k=2)
        self.mock_model.output_file_names["metric_plot"][target_name] = output_file
        self.mock_model.contingency_tables[target_name] = pl.DataFrame(
            {
                "k": [1, 1, 1, 2, 2, 2],
                "label": [0, 1, 0, 0, 1, 1],
                "score": [0.1, 0.9, 0.2, 0.3, 0.8, 0.7],
            }
        )

        create_metric_plots(self.mock_model)

        # Assert file exists and has size > 0
        self.assertTrue(os.path.exists(output_file))
        self.assertGreater(os.path.getsize(output_file), 0)

    def test_missing_classes_in_fold(self) -> None:
        """
        Verify that the function handles edge cases where a specific fold might
        only contain one class (which causes roc_curve to fail if not skipped).
        """
        target_name = "pres"
        output_file = os.path.join(self.test_dir, f"plot_{target_name}.svg")

        self.mock_model.output_file_names["metric_plot"][target_name] = output_file
        self.mock_model.contingency_tables[target_name] = pl.DataFrame(
            {
                # k=1 has both classes (OK)
                # k=2 has only class 0 (Should be skipped by logic)
                "k": [1, 1, 2, 2],
                "label": [0, 1, 0, 0],
                "score": [0.1, 0.9, 0.2, 0.3],
            }
        )

        # Should not raise an error
        create_metric_plots(self.mock_model)
        self.assertTrue(os.path.exists(output_file))
