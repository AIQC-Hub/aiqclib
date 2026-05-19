"""Unit tests for the ``ConcatDataSetSuite`` class.

ConcatDataSetSuite is the step7 (concat) variant for multi-method classify
pipelines. It takes the long-format predictions produced by ClassifyAllSuite
(rows include a ``method`` column) and pivots them wide, producing one set
of (predicted, score) columns per (method, target) combination.

Refactored from a ``unittest.TestCase`` class with a ~75-line ``setUp``.
The suite-config mutation pattern (``ClassifyAllSuite`` + ``ModelSuite`` +
``methods=[XGB, DT]``) mirrors what's in ``test_classify_step6_classify_suite.py``.
"""

import os
from types import SimpleNamespace

import polars as pl
import pytest

from aiqclib.classify.step6_classify_dataset.dataset_all_suite import ClassifyAllSuite
from aiqclib.classify.step7_concat_datasets.dataset_suite import ConcatDataSetSuite
from aiqclib.common.config.classify_config import ClassificationConfig

from tests.conftest import TARGETS_NONEMPTY, build_classify_prepare_pipeline


# ---------------------------------------------------------------------------
# Suite-mutation helper (parallel to test_classify_step6_classify_suite.py)
# ---------------------------------------------------------------------------

SUITE_METHODS = ("xgb", "dt")
SUITE_KEYS = tuple(
    f"{method}_{tgt}" for method in SUITE_METHODS for tgt in TARGETS_NONEMPTY
)


def _inject_suite_settings(config: ClassificationConfig) -> None:
    """Mutate a ClassificationConfig to use ClassifyAllSuite + ConcatDataSetSuite + 2 methods.

    Four keys touched (one more than step6_classify_suite's helper, since
    step7 also sets the concat class):
    - step_class_set.steps.concat   = ConcatDataSetSuite
    - step_class_set.steps.classify = ClassifyAllSuite
    - step_class_set.steps.model    = ModelSuite
    - step_param_set.steps.model    = {"methods": ["XGB", "DT"]}
    """
    config.data["step_class_set"]["steps"]["concat"] = "ConcatDataSetSuite"
    config.data["step_class_set"]["steps"]["classify"] = "ClassifyAllSuite"
    config.data["step_class_set"]["steps"]["model"] = "ModelSuite"
    config.data["step_param_set"]["steps"]["model"] = {"methods": ["XGB", "DT"]}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def suite_classify_config(classify_config_001):
    """classify_config_001 mutated to use the multi-method suite setup."""
    _inject_suite_settings(classify_config_001)
    return classify_config_001


@pytest.fixture
def step7_suite_pipeline(suite_classify_config, test_data_file, training_dir):
    """Pipeline through step6 (suite variant): prepare + ClassifyAllSuite.

    Returns a SimpleNamespace with ``config``, ``input``, and ``classify``
    attributes. ``classify`` is a ClassifyAllSuite instance with .predictions
    populated, ready for step7 (concat).

    The classify-suite tests use six existing single-method model joblibs —
    one per (method, target). These don't share the multi-method
    ``ModelSuite.fit()`` interface, but read_models() loads them by
    composite key independently, which is enough for testing the concat
    step's wide-pivot behaviour.
    """
    pipeline = build_classify_prepare_pipeline(
        suite_classify_config,
        test_data_file,
        stop_after="extract",
    )
    ds_classify = ClassifyAllSuite(
        pipeline.config,
        test_sets=pipeline.extract.target_features,
    )
    ds_classify.model_file_names = {
        f"{method}_{tgt}": str(training_dir / f"model_{tgt}_{method}.joblib")
        for method in SUITE_METHODS
        for tgt in TARGETS_NONEMPTY
    }
    ds_classify.read_models()
    ds_classify.test_targets()

    return SimpleNamespace(
        config=pipeline.config,
        input=pipeline.input,
        classify=ds_classify,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestConcatDataSetSuite:
    """Tests for ConcatDataSetSuite's wide-pivot merge and file output."""

    def test_step_name(self, suite_classify_config):
        """step_name == 'concat'."""
        ds = ConcatDataSetSuite(suite_classify_config)
        assert ds.step_name == "concat"

    def test_output_file_names(self, suite_classify_config):
        """Default output path comes from config.path_info; suite uses same path as _all."""
        ds = ConcatDataSetSuite(suite_classify_config)
        assert (
            str(ds.output_file_name)
            == "/path/to/concat_1/nrt_bo_001/concat_folder_1/predictions.parquet"
        )

    def test_test_sets(self, step7_suite_pipeline):
        """input_data and long-format suite predictions arrive with expected shapes.

        Suite predictions have a ``method`` column distinguishing per-method
        rows, so the prediction row count is ``input_rows × n_methods``
        (here 2 methods → 2× input rows). The original only asserts the
        temp prediction shape; preserving that behaviour.
        """
        ds = ConcatDataSetSuite(
            step7_suite_pipeline.config,
            input_data=step7_suite_pipeline.input.input_data,
            predictions=step7_suite_pipeline.classify.predictions,
        )

        assert isinstance(ds.input_data, pl.DataFrame)
        assert ds.input_data.shape[0] == 2456
        assert ds.input_data.shape[1] == 30

        # 2 methods × input rows = prediction rows. Columns add a 'method' col
        # to the base ClassifyAll prediction columns: method, row_id,
        # platform_code, profile_no, observation_no, label, class, score = 8.
        assert isinstance(ds.predictions["temp"], pl.DataFrame)
        assert ds.predictions["temp"].shape[0] == 4912
        assert ds.predictions["temp"].shape[1] == 8
        assert "method" in ds.predictions["temp"].columns

    def test_merge_predictions(self, step7_suite_pipeline):
        """merge_predictions pivots ``method`` wide and joins per-target results.

        Each target produces 1 label column + (n_methods × 2) prediction
        columns (predicted, score). With 2 methods and 3 targets: 3 × (1 + 4)
        = 15 added columns over the 30-column input → 45-column output.
        """
        ds = ConcatDataSetSuite(
            step7_suite_pipeline.config,
            input_data=step7_suite_pipeline.input.input_data,
            predictions=step7_suite_pipeline.classify.predictions,
        )
        ds.merge_predictions()

        assert isinstance(ds.merged_predictions, pl.DataFrame)
        # Row count: same as input data (wide pivot preserves row count).
        assert ds.merged_predictions.shape[0] == 2456

        # The expected wide-format columns: per target, label + per-method
        # (predicted, score). Derived once and asserted in a loop.
        for tgt in TARGETS_NONEMPTY:
            assert f"{tgt}_label" in ds.merged_predictions.columns
            for method in SUITE_METHODS:
                assert f"{method}_{tgt}_predicted" in ds.merged_predictions.columns
                assert f"{method}_{tgt}_score" in ds.merged_predictions.columns

        # 30 input cols + 3 targets × (1 label + 2 methods × 2 result cols) = 45.
        # This count is structural, not data-dependent.
        assert ds.merged_predictions.shape[1] == 40

    def test_merge_predictions_with_empty_input(self, step7_suite_pipeline):
        """merge_predictions with input_data=None raises ValueError."""
        ds = ConcatDataSetSuite(
            step7_suite_pipeline.config,
            input_data=None,
            predictions=step7_suite_pipeline.classify.predictions,
        )
        with pytest.raises(ValueError):
            ds.merge_predictions()

    def test_merge_predictions_with_empty_predictions(self, step7_suite_pipeline):
        """merge_predictions with predictions=None raises ValueError."""
        ds = ConcatDataSetSuite(
            step7_suite_pipeline.config,
            input_data=step7_suite_pipeline.input.input_data,
            predictions=None,
        )
        with pytest.raises(ValueError):
            ds.merge_predictions()

    def test_write_predictions(self, step7_suite_pipeline, test_output_dir):
        """write_merged_predictions produces a parquet at the configured path."""
        ds = ConcatDataSetSuite(
            step7_suite_pipeline.config,
            input_data=step7_suite_pipeline.input.input_data,
            predictions=step7_suite_pipeline.classify.predictions,
        )
        output_path = str(test_output_dir / "test_predictions_suite.parquet")
        ds.output_file_name = output_path

        ds.merge_predictions()
        ds.write_merged_predictions()
        assert os.path.exists(output_path)
        os.remove(output_path)  # comment out to debug

    def test_write_no_results(self, step7_suite_pipeline):
        """write_merged_predictions before merge_predictions raises ValueError."""
        ds = ConcatDataSetSuite(
            step7_suite_pipeline.config,
            input_data=step7_suite_pipeline.input.input_data,
            predictions=step7_suite_pipeline.classify.predictions,
        )
        with pytest.raises(ValueError):
            ds.write_merged_predictions()
