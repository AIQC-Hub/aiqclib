"""Unit tests for the ``create_metric_plots`` utility.

create_metric_plots takes a model-like object exposing ``contingency_tables``
and ``output_file_names["metric_plot"]`` and writes ROC + Precision-Recall
plots to disk as SVG files. The tests verify:
- Empty contingency_tables raises ValueError
- Single-fold (test-set) data produces a valid SVG file
- Multi-fold (cross-validation) data also produces a valid SVG, exercising
  the mean-curve + std-deviation code path
- A fold containing only one class is silently skipped (instead of crashing
  ``roc_curve``)

Refactored from a ``unittest.TestCase`` class with tempfile.mkdtemp +
shutil.rmtree teardown. Now uses:
- ``test_output_dir`` from conftest (real directory under tests/data/test/);
  ``os.remove(...)  # comment out to debug`` after each assertion lets the
  generated SVG be inspected on failure.
- A ``mock_model`` fixture for per-test isolation of the MockModel state.

The MockModel class stays at module level — test infrastructure, not data.
"""

import os
from typing import Dict

import matplotlib
import polars as pl
import pytest

# Non-interactive backend so plot tests don't open windows. Must be set
# before any aiqclib import that loads matplotlib's pyplot. Keep this line
# directly above the create_metric_plots import.
matplotlib.use("Agg")

from aiqclib.common.utils.metric_plots import create_metric_plots


# ---------------------------------------------------------------------------
# Module-level mock
# ---------------------------------------------------------------------------

class MockModel:
    """Minimal stand-in for ValidationBase / BuildModelBase.

    create_metric_plots only reads two attributes: ``contingency_tables``
    (per-target DataFrame) and ``output_file_names["metric_plot"]``
    (per-target path). This mock supplies both without dragging in any
    of the real wrapper classes.
    """

    def __init__(self) -> None:
        self.contingency_tables: Dict[str, pl.DataFrame] = {}
        self.output_file_names: Dict[str, Dict[str, str]] = {"metric_plot": {}}


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_model():
    """Fresh MockModel per test — avoids contingency_tables leaking across tests."""
    return MockModel()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCreateMetricPlots:
    """Tests for create_metric_plots's output-file generation behaviour."""

    def test_empty_contingency_tables(self, mock_model):
        """create_metric_plots with no contingency_tables raises ValueError."""
        mock_model.contingency_tables = {}
        with pytest.raises(ValueError):
            create_metric_plots(mock_model)

    def test_single_fold_plot_generation(self, mock_model, test_output_dir):
        """A single-fold contingency table (e.g. test set with k=1) produces an SVG.

        With one fold there's no mean/std logic — just the single ROC/PR
        curve. The output file must exist and have non-zero size.
        """
        target_name = "temp"
        output_path = str(test_output_dir / f"test_metric_plot_{target_name}.svg")
        mock_model.output_file_names["metric_plot"][target_name] = output_path
        mock_model.contingency_tables[target_name] = pl.DataFrame({
            "k": [1, 1, 1, 1, 1],
            "label": [0, 0, 1, 1, 0],
            "score": [0.1, 0.2, 0.8, 0.9, 0.4],
        })

        create_metric_plots(mock_model)

        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0
        os.remove(output_path)  # comment out to debug

    def test_multi_fold_plot_generation(self, mock_model, test_output_dir):
        """Multi-fold contingency tables exercise the mean-curve + std-dev logic.

        Two folds (k=1, k=2) means create_metric_plots computes per-fold
        ROC/PR curves, then averages them with a confidence band. The
        output file must exist and have non-zero size.
        """
        target_name = "psal"
        output_path = str(test_output_dir / f"test_metric_plot_{target_name}.svg")
        mock_model.output_file_names["metric_plot"][target_name] = output_path
        mock_model.contingency_tables[target_name] = pl.DataFrame({
            "k": [1, 1, 1, 2, 2, 2],
            "label": [0, 1, 0, 0, 1, 1],
            "score": [0.1, 0.9, 0.2, 0.3, 0.8, 0.7],
        })

        create_metric_plots(mock_model)

        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0
        os.remove(output_path)  # comment out to debug

    def test_missing_classes_in_fold(self, mock_model, test_output_dir):
        """A fold containing only one class is silently skipped.

        sklearn's ``roc_curve`` errors when called on single-class data.
        create_metric_plots should detect this and skip the problematic
        fold instead of crashing — the test verifies success by checking
        that the output file gets created (using k=1 which has both classes).
        """
        target_name = "pres"
        output_path = str(test_output_dir / f"test_metric_plot_{target_name}.svg")
        mock_model.output_file_names["metric_plot"][target_name] = output_path
        mock_model.contingency_tables[target_name] = pl.DataFrame({
            # k=1 has both classes; k=2 has only class 0 (must be skipped).
            "k": [1, 1, 2, 2],
            "label": [0, 1, 0, 0],
            "score": [0.1, 0.9, 0.2, 0.3],
        })

        # Should not raise.
        create_metric_plots(mock_model)

        assert os.path.exists(output_path)
        os.remove(output_path)  # comment out to debug