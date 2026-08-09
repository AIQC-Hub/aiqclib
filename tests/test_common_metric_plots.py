"""Unit tests for the ``create_metric_plots`` utility.

create_metric_plots takes a model-like object exposing ``model_scores``
and ``output_file_names["metric_plot"]`` and writes ROC + Precision-Recall
plots to disk as SVG files. The tests verify:
- Empty model_scores raises ValueError
- Single-fold (test-set) data produces a valid SVG file
- Multi-fold (cross-validation) data also produces a valid SVG, exercising
  the mean-curve + std-deviation code path
- A fold containing only one class is silently skipped (instead of crashing
  ``roc_curve``)
- When no fold has both classes there is no curve to label, so the plot
  explains itself instead of making matplotlib warn about an empty legend

Refactored from a ``unittest.TestCase`` class with tempfile.mkdtemp +
shutil.rmtree teardown. Now uses:
- ``test_output_dir`` from conftest (real directory under tests/data/test/);
  ``os.remove(...)  # comment out to debug`` after each assertion lets the
  generated SVG be inspected on failure.
- A ``mock_model`` fixture for per-test isolation of the MockModel state.

The MockModel class stays at module level — test infrastructure, not data.
"""

import os
import warnings
from typing import Dict

import matplotlib
import polars as pl
import pytest

# Non-interactive backend so plot tests don't open windows. Must be set
# before any aiqclib import that loads matplotlib's pyplot. Keep this line
# directly above the create_metric_plots import.
matplotlib.use("Agg")

from aiqclib.common.utils.metric_plots import (
    EMPTY_PLOT_NOTE,
    create_metric_plots,
    create_multi_method_metric_plots,
)


# ---------------------------------------------------------------------------
# Module-level mock
# ---------------------------------------------------------------------------


class MockModel:
    """Minimal stand-in for ValidationBase / BuildModelBase.

    create_metric_plots only reads two attributes: ``model_scores``
    (per-target DataFrame) and ``output_file_names["metric_plot"]``
    (per-target path). This mock supplies both without dragging in any
    of the real wrapper classes.
    """

    def __init__(self) -> None:
        self.model_scores: Dict[str, pl.DataFrame] = {}
        self.output_file_names: Dict[str, Dict[str, str]] = {"metric_plot": {}}


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_model():
    """Fresh MockModel per test — avoids model_scores leaking across tests."""
    return MockModel()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCreateMetricPlots:
    """Tests for create_metric_plots's output-file generation behaviour."""

    def test_empty_model_scores(self, mock_model):
        """create_metric_plots with no model_scores raises ValueError."""
        mock_model.model_scores = {}
        with pytest.raises(ValueError):
            create_metric_plots(mock_model)

    def test_single_fold_plot_generation(self, mock_model, test_output_dir):
        """A single-fold model-scores table (e.g. test set with k=1) produces an SVG.

        With one fold there's no mean/std logic — just the single ROC/PR
        curve. The output file must exist and have non-zero size.
        """
        target_name = "temp"
        output_path = str(test_output_dir / f"test_metric_plot_{target_name}.svg")
        mock_model.output_file_names["metric_plot"][target_name] = output_path
        mock_model.model_scores[target_name] = pl.DataFrame(
            {
                "k": [1, 1, 1, 1, 1],
                "label": [0, 0, 1, 1, 0],
                "score": [0.1, 0.2, 0.8, 0.9, 0.4],
            }
        )

        create_metric_plots(mock_model)

        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0
        os.remove(output_path)  # comment out to debug

    def test_multi_fold_plot_generation(self, mock_model, test_output_dir):
        """Multi-fold model-scores tables exercise the mean-curve + std-dev logic.

        Two folds (k=1, k=2) means create_metric_plots computes per-fold
        ROC/PR curves, then averages them with a confidence band. The
        output file must exist and have non-zero size.
        """
        target_name = "psal"
        output_path = str(test_output_dir / f"test_metric_plot_{target_name}.svg")
        mock_model.output_file_names["metric_plot"][target_name] = output_path
        mock_model.model_scores[target_name] = pl.DataFrame(
            {
                "k": [1, 1, 1, 2, 2, 2],
                "label": [0, 1, 0, 0, 1, 1],
                "score": [0.1, 0.9, 0.2, 0.3, 0.8, 0.7],
            }
        )

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
        mock_model.model_scores[target_name] = pl.DataFrame(
            {
                # k=1 has both classes; k=2 has only class 0 (must be skipped).
                "k": [1, 1, 2, 2],
                "label": [0, 1, 0, 0],
                "score": [0.1, 0.9, 0.2, 0.3],
            }
        )

        # Should not raise.
        create_metric_plots(mock_model)

        assert os.path.exists(output_path)
        os.remove(output_path)  # comment out to debug


class TestSingleClassPlots:
    """Plots for an evaluation whose labels hold a single class."""

    @staticmethod
    def _single_class_scores() -> pl.DataFrame:
        """Scores whose labels are all class 0, as a label-starved run produces."""
        return pl.DataFrame(
            {
                "k": [0, 0, 0, 0],
                "label": [0, 0, 0, 0],
                "score": [0.1, 0.2, 0.3, 0.4],
            }
        )

    def test_no_matplotlib_legend_warning(self, mock_model, test_output_dir):
        """The empty legend must not produce matplotlib's unhelpful warning."""
        target_name = "temp"
        output_path = str(test_output_dir / f"test_single_class_{target_name}.svg")
        mock_model.output_file_names["metric_plot"][target_name] = output_path
        mock_model.model_scores[target_name] = self._single_class_scores()

        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning fails the test
            create_metric_plots(mock_model)

        assert os.path.exists(output_path)
        os.remove(output_path)  # comment out to debug

    def test_plot_explains_why_it_is_empty(self, mock_model, test_output_dir):
        """The saved SVG carries a note instead of silently showing blank axes."""
        target_name = "temp"
        output_path = str(test_output_dir / f"test_single_class_note_{target_name}.svg")
        mock_model.output_file_names["metric_plot"][target_name] = output_path
        mock_model.model_scores[target_name] = self._single_class_scores()

        create_metric_plots(mock_model)

        svg = open(output_path, encoding="utf-8").read()
        # The note is rendered as text spans; check its distinctive first word
        # survives into the SVG rather than matching the embedded newline.
        assert "single class" in svg.replace("\n", " ") or "single" in svg
        os.remove(output_path)  # comment out to debug

    def test_multi_method_plots_are_also_quiet(self, mock_model, test_output_dir):
        """The multi-method variant shares the same empty-legend handling."""
        target_name = "psal"
        output_path = str(
            test_output_dir / f"test_single_class_multi_{target_name}.svg"
        )
        mock_model.output_file_names["metric_plot"][target_name] = output_path
        mock_model.model_scores[target_name] = pl.DataFrame(
            {
                "method": ["XGB", "XGB", "RF", "RF"],
                "label": [0, 0, 0, 0],
                "score": [0.1, 0.2, 0.3, 0.4],
            }
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            create_multi_method_metric_plots(mock_model)

        assert os.path.exists(output_path)
        os.remove(output_path)  # comment out to debug

    def test_normal_data_still_gets_a_legend(self, mock_model, test_output_dir):
        """Two-class data keeps its legend; the note is not added."""
        target_name = "temp"
        output_path = str(test_output_dir / f"test_two_class_{target_name}.svg")
        mock_model.output_file_names["metric_plot"][target_name] = output_path
        mock_model.model_scores[target_name] = pl.DataFrame(
            {
                "k": [0, 0, 0, 0],
                "label": [0, 1, 0, 1],
                "score": [0.1, 0.9, 0.2, 0.8],
            }
        )

        create_metric_plots(mock_model)

        svg = open(output_path, encoding="utf-8").read()
        assert EMPTY_PLOT_NOTE.split("\n")[0] not in svg
        os.remove(output_path)  # comment out to debug
