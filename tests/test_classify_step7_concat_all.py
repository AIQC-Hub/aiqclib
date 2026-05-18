"""Unit tests for the ``ConcatDataSetAll`` class.

ConcatDataSetAll is the step7 (concat) variant for single-method classify
pipelines. It takes the predictions produced by ClassifyAll (one prediction
DataFrame per target) and merges them with the original input data into a
single long parquet output.

Refactored from a ``unittest.TestCase`` class with a ~70-line ``setUp``
that ran the full classify pipeline + step6 in-line. The pipeline setup
moves to a per-file fixture using ``build_classify_prepare_pipeline()``
from conftest; the step6 classify step is local to this file since it's
specific to step7's needs.
"""

import os
from types import SimpleNamespace

import polars as pl
import pytest

from aiqclib.classify.step6_classify_dataset.dataset_all import ClassifyAll
from aiqclib.classify.step7_concat_datasets.dataset_all import ConcatDataSetAll

from tests.conftest import TARGETS_NONEMPTY, build_classify_prepare_pipeline


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def step7_pipeline(classify_config_001, test_data_file, training_dir):
    """Pipeline through step6: prepare (1-5) + classify (6) with default models.

    Returns a SimpleNamespace with ``config``, ``input``, and ``classify``
    attributes. ``classify`` is a ClassifyAll instance with .predictions
    populated, ready for step7 (concat).
    """
    pipeline = build_classify_prepare_pipeline(
        classify_config_001, test_data_file, stop_after="extract",
    )
    ds_classify = ClassifyAll(
        pipeline.config, test_sets=pipeline.extract.target_features,
    )
    ds_classify.model_file_names = {
        tgt: str(training_dir / f"model_{tgt}.joblib") for tgt in TARGETS_NONEMPTY
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

class TestConcatPredictions:
    """Tests for ConcatDataSetAll's merge_predictions and file output."""

    def test_step_name(self, classify_config_001):
        """step_name == 'concat'."""
        ds = ConcatDataSetAll(classify_config_001)
        assert ds.step_name == "concat"

    def test_output_file_names(self, classify_config_001):
        """Default output path comes from config.path_info."""
        ds = ConcatDataSetAll(classify_config_001)
        assert (
            str(ds.output_file_name)
            == "/path/to/concat_1/nrt_bo_001/concat_folder_1/predictions.parquet"
        )

    def test_test_sets(self, step7_pipeline):
        """input_data and per-target predictions are loaded with expected shapes."""
        ds = ConcatDataSetAll(
            step7_pipeline.config,
            input_data=step7_pipeline.input.input_data,
            predictions=step7_pipeline.classify.predictions,
        )

        assert isinstance(ds.input_data, pl.DataFrame)
        assert ds.input_data.shape[0] == 2456
        assert ds.input_data.shape[1] == 30

        # Each target's predictions table is (input_rows × 7).
        for tgt in TARGETS_NONEMPTY:
            assert isinstance(ds.predictions[tgt], pl.DataFrame)
            assert ds.predictions[tgt].shape[0] == 2456
            assert ds.predictions[tgt].shape[1] == 7

    def test_merge_predictions(self, step7_pipeline):
        """merge_predictions combines input + per-target predictions into one wide frame."""
        ds = ConcatDataSetAll(
            step7_pipeline.config,
            input_data=step7_pipeline.input.input_data,
            predictions=step7_pipeline.classify.predictions,
        )
        ds.merge_predictions()

        assert isinstance(ds.merged_predictions, pl.DataFrame)
        # Row count == input rows; column count == 30 input cols + 3 × 3 added per-target cols.
        assert ds.merged_predictions.shape[0] == 2456
        assert ds.merged_predictions.shape[1] == 36

    def test_merge_predictions_with_empty_input(self, step7_pipeline):
        """merge_predictions with input_data=None raises ValueError."""
        ds = ConcatDataSetAll(
            step7_pipeline.config,
            input_data=None,
            predictions=step7_pipeline.classify.predictions,
        )
        with pytest.raises(ValueError):
            ds.merge_predictions()

    def test_merge_predictions_with_empty_predictions(self, step7_pipeline):
        """merge_predictions with predictions=None raises ValueError."""
        ds = ConcatDataSetAll(
            step7_pipeline.config,
            input_data=step7_pipeline.input.input_data,
            predictions=None,
        )
        with pytest.raises(ValueError):
            ds.merge_predictions()

    def test_write_predictions(self, step7_pipeline, test_output_dir):
        """write_merged_predictions produces a parquet at the configured path."""
        ds = ConcatDataSetAll(
            step7_pipeline.config,
            input_data=step7_pipeline.input.input_data,
            predictions=step7_pipeline.classify.predictions,
        )
        output_path = str(test_output_dir / "test_predictions.parquet")
        ds.output_file_name = output_path

        ds.merge_predictions()
        ds.write_merged_predictions()
        assert os.path.exists(output_path)
        os.remove(output_path)  # comment out to debug

    def test_write_no_results(self, step7_pipeline):
        """write_merged_predictions before merge_predictions raises ValueError."""
        ds = ConcatDataSetAll(
            step7_pipeline.config,
            input_data=step7_pipeline.input.input_data,
            predictions=step7_pipeline.classify.predictions,
        )
        with pytest.raises(ValueError):
            ds.write_merged_predictions()