"""Unit tests for the classify-stage dataset loader functions.

Seven classes, one per classify-pipeline step:
- ``TestClassifyInputClassLoader``     — load_classify_step1_input_dataset
- ``TestClassifySummaryClassLoader``   — load_classify_step2_summary_dataset
- ``TestClassifySelectClassLoader``    — load_classify_step3_select_dataset
- ``TestClassifyLocateClassLoader``    — load_classify_step4_locate_dataset
- ``TestClassifyExtractClassLoader``   — load_classify_step5_extract_dataset
- ``TestClassifyClassifyClassLoader``  — load_classify_step6_classify_dataset
- ``TestClassifyConcatClassLoader``    — load_classify_step7_concat_dataset

For each loader:
1. The default config produces an instance of the expected class with the
   right step_name.
2. Provided upstream outputs propagate to the resulting wrapper with
   expected shapes.

Steps 6 and 7 also cover the suite variants (ClassifyAllSuite,
ConcatDataSetSuite), which require config mutations (model=ModelSuite,
classify=ClassifyAllSuite, concat=ConcatDataSetSuite).

Refactored from 7 ``unittest.TestCase`` classes. The setUp pattern (load
test_classify_001.yaml + select NRT_BO_001 + define test_data_file path)
is replaced with ``classify_config_001`` + ``test_data_file`` fixtures
from conftest. The "build full prepare pipeline" pattern (run steps 1-N
to set up the input to step N+1) is replaced with
``build_classify_prepare_pipeline`` from conftest.

Note on suite tests in step 7:
The concat suite test sets four config keys (concat/classify/model + the
methods param list). Same pattern as ``test_classify_step7_concat_suite.py``.
"""

import polars as pl
import pytest

from aiqclib.classify.step1_read_input.dataset_all import InputDataSetAll
from aiqclib.classify.step2_calc_stats.dataset_all import SummaryDataSetAll
from aiqclib.classify.step3_select_profiles.dataset_all import SelectDataSetAll
from aiqclib.classify.step4_select_rows.dataset_all import LocateDataSetAll
from aiqclib.classify.step5_extract_features.dataset_all import ExtractDataSetAll
from aiqclib.classify.step6_classify_dataset.dataset_all import ClassifyAll
from aiqclib.classify.step6_classify_dataset.dataset_all_suite import ClassifyAllSuite
from aiqclib.classify.step7_concat_datasets.dataset_all import ConcatDataSetAll
from aiqclib.classify.step7_concat_datasets.dataset_suite import ConcatDataSetSuite
from aiqclib.common.loader.classify_loader import (
    load_classify_step1_input_dataset,
    load_classify_step2_summary_dataset,
    load_classify_step3_select_dataset,
    load_classify_step4_locate_dataset,
    load_classify_step5_extract_dataset,
    load_classify_step6_classify_dataset,
    load_classify_step7_concat_dataset,
)

from tests.conftest import TARGETS_NONEMPTY, build_classify_prepare_pipeline


# ---------------------------------------------------------------------------
# Shared shape expectations
# ---------------------------------------------------------------------------

_CLASSIFY_INPUT_ROWS = 2456
_CLASSIFY_INPUT_COLS = 30
_CLASSIFY_SUMMARY_ROWS = 44
_CLASSIFY_SUMMARY_COLS = 12
_CLASSIFY_PROFILE_COUNT = 10


# ---------------------------------------------------------------------------
# Step 1: input
# ---------------------------------------------------------------------------

class TestClassifyInputClassLoader:
    """Tests for load_classify_step1_input_dataset."""

    def test_load_dataset_valid_config(self, classify_config_001):
        """Default config produces an InputDataSetAll with step_name='input'."""
        ds = load_classify_step1_input_dataset(classify_config_001)
        assert isinstance(ds, InputDataSetAll)
        assert ds.step_name == "input"

    def test_load_input_class_with_invalid_config(self, classify_config_001):
        """An invalid input-class name raises ValueError."""
        classify_config_001.data["step_class_set"]["steps"]["input"] = "InvalidClass"
        with pytest.raises(ValueError):
            _ = load_classify_step1_input_dataset(classify_config_001)


# ---------------------------------------------------------------------------
# Step 2: summary
# ---------------------------------------------------------------------------

class TestClassifySummaryClassLoader:
    """Tests for load_classify_step2_summary_dataset."""

    def test_load_dataset_valid_config(self, classify_config_001):
        """Default config produces a SummaryDataSetAll with step_name='summary'."""
        ds = load_classify_step2_summary_dataset(classify_config_001)
        assert isinstance(ds, SummaryDataSetAll)
        assert ds.step_name == "summary"

    def test_load_dataset_input_data(self, classify_config_001, classify_input_001):
        """Provided input_data propagates with the expected shape."""
        ds = load_classify_step2_summary_dataset(
            classify_config_001, classify_input_001.input_data,
        )
        assert isinstance(ds, SummaryDataSetAll)
        assert isinstance(ds.input_data, pl.DataFrame)
        assert ds.input_data.shape[0] == _CLASSIFY_INPUT_ROWS
        assert ds.input_data.shape[1] == _CLASSIFY_INPUT_COLS


# ---------------------------------------------------------------------------
# Step 3: select
# ---------------------------------------------------------------------------

class TestClassifySelectClassLoader:
    """Tests for load_classify_step3_select_dataset."""

    def test_load_dataset_valid_config(self, classify_config_001):
        """Default config produces a SelectDataSetAll with step_name='select'."""
        ds = load_classify_step3_select_dataset(classify_config_001)
        assert isinstance(ds, SelectDataSetAll)
        assert ds.step_name == "select"

    def test_load_dataset_input_data(self, classify_config_001, classify_input_001):
        """Provided input_data propagates with the expected shape."""
        ds = load_classify_step3_select_dataset(
            classify_config_001, classify_input_001.input_data,
        )
        assert isinstance(ds, SelectDataSetAll)
        assert isinstance(ds.input_data, pl.DataFrame)
        assert ds.input_data.shape[0] == _CLASSIFY_INPUT_ROWS
        assert ds.input_data.shape[1] == _CLASSIFY_INPUT_COLS


# ---------------------------------------------------------------------------
# Step 4: locate
# ---------------------------------------------------------------------------

class TestClassifyLocateClassLoader:
    """Tests for load_classify_step4_locate_dataset."""

    def test_load_dataset_valid_config(self, classify_config_001):
        """Default config produces a LocateDataSetAll with step_name='locate'."""
        ds = load_classify_step4_locate_dataset(classify_config_001)
        assert isinstance(ds, LocateDataSetAll)
        assert ds.step_name == "locate"

    def test_load_dataset_input_data_and_profiles(
        self, classify_config_001, classify_input_001, classify_select_001,
    ):
        """Provided input_data + selected_profiles propagate with expected shapes."""
        ds = load_classify_step4_locate_dataset(
            classify_config_001,
            classify_input_001.input_data,
            classify_select_001.selected_profiles,
        )

        assert isinstance(ds, LocateDataSetAll)

        assert isinstance(ds.input_data, pl.DataFrame)
        assert ds.input_data.shape[0] == _CLASSIFY_INPUT_ROWS
        assert ds.input_data.shape[1] == _CLASSIFY_INPUT_COLS

        assert isinstance(ds.selected_profiles, pl.DataFrame)
        assert ds.selected_profiles.shape[0] == _CLASSIFY_PROFILE_COUNT
        assert ds.selected_profiles.shape[1] == 8


# ---------------------------------------------------------------------------
# Step 5: extract
# ---------------------------------------------------------------------------

class TestClassifyExtractClassLoader:
    """Tests for load_classify_step5_extract_dataset."""

    def test_load_dataset_valid_config(self, classify_config_001):
        """Default config produces an ExtractDataSetAll with step_name='extract'."""
        ds = load_classify_step5_extract_dataset(classify_config_001)
        assert isinstance(ds, ExtractDataSetAll)
        assert ds.step_name == "extract"

    def test_load_dataset_input_data_and_profiles(
        self, classify_config_001, test_data_file,
    ):
        """Provided upstream outputs (input/summary/select/locate) propagate correctly.

        Uses ``build_classify_prepare_pipeline(stop_after="locate")`` to build
        the four upstream stages, then calls load_classify_step5_extract_dataset
        directly with the assembled inputs.
        """
        pipeline = build_classify_prepare_pipeline(
            classify_config_001, test_data_file, stop_after="locate",
        )

        ds = load_classify_step5_extract_dataset(
            classify_config_001,
            pipeline.input.input_data,
            pipeline.select.selected_profiles,
            pipeline.locate.selected_rows,
            pipeline.summary.summary_stats,
        )

        assert isinstance(ds, ExtractDataSetAll)

        assert isinstance(ds.input_data, pl.DataFrame)
        assert ds.input_data.shape[0] == _CLASSIFY_INPUT_ROWS
        assert ds.input_data.shape[1] == _CLASSIFY_INPUT_COLS

        assert isinstance(ds.summary_stats, pl.DataFrame)
        assert ds.summary_stats.shape[0] == _CLASSIFY_SUMMARY_ROWS
        assert ds.summary_stats.shape[1] == _CLASSIFY_SUMMARY_COLS

        assert isinstance(ds.selected_profiles, pl.DataFrame)
        assert ds.selected_profiles.shape[0] == _CLASSIFY_PROFILE_COUNT
        assert ds.selected_profiles.shape[1] == 8

        # filtered_input matches input for the classify-side "all" pipeline.
        assert isinstance(ds.filtered_input, pl.DataFrame)
        assert ds.filtered_input.shape[0] == _CLASSIFY_INPUT_ROWS
        assert ds.filtered_input.shape[1] == _CLASSIFY_INPUT_COLS

        # Each target keeps all input rows (the "all" loaders don't filter
        # by QC values like the prepare-side LocateDataSetA does).
        for tgt in ("temp", "psal"):
            assert isinstance(ds.selected_rows[tgt], pl.DataFrame)
            assert ds.selected_rows[tgt].shape[0] == _CLASSIFY_INPUT_ROWS
            assert ds.selected_rows[tgt].shape[1] == 9


# ---------------------------------------------------------------------------
# Step 6: classify
# ---------------------------------------------------------------------------

def _inject_suite_classify_settings(config) -> None:
    """Mutate config to use ClassifyAllSuite + ModelSuite."""
    config.data["step_class_set"]["steps"]["model"] = "ModelSuite"
    config.data["step_class_set"]["steps"]["classify"] = "ClassifyAllSuite"


class TestClassifyClassifyClassLoader:
    """Tests for load_classify_step6_classify_dataset (default + suite variants)."""

    def test_load_dataset_valid_config(self, classify_config_001):
        """Default config produces a ClassifyAll with step_name='classify'."""
        ds = load_classify_step6_classify_dataset(classify_config_001)
        assert isinstance(ds, ClassifyAll)
        assert ds.step_name == "classify"

    def test_load_classifiction_suite_dataset(self, classify_config_001):
        """Suite config produces a ClassifyAllSuite with step_name='classify'."""
        _inject_suite_classify_settings(classify_config_001)
        ds = load_classify_step6_classify_dataset(classify_config_001)
        assert isinstance(ds, ClassifyAllSuite)
        assert ds.step_name == "classify"

    def test_load_dataset_input_data(self, classify_config_001, test_data_file):
        """Provided target_features (from step5) populate test_sets per target."""
        pipeline = build_classify_prepare_pipeline(
            classify_config_001, test_data_file, stop_after="extract",
        )

        ds = load_classify_step6_classify_dataset(
            classify_config_001, pipeline.extract.target_features,
        )

        assert isinstance(ds, ClassifyAll)
        for tgt in ("temp", "psal"):
            assert isinstance(ds.test_sets[tgt], pl.DataFrame)
            assert ds.test_sets[tgt].shape[0] == _CLASSIFY_INPUT_ROWS
            assert ds.test_sets[tgt].shape[1] == 56

    def test_load_suite_dataset_input_data(self, classify_config_001, test_data_file):
        """Same as test_load_dataset_input_data but for the suite variant."""
        _inject_suite_classify_settings(classify_config_001)
        pipeline = build_classify_prepare_pipeline(
            classify_config_001, test_data_file, stop_after="extract",
        )

        ds = load_classify_step6_classify_dataset(
            classify_config_001, pipeline.extract.target_features,
        )

        assert isinstance(ds, ClassifyAllSuite)
        for tgt in ("temp", "psal"):
            assert isinstance(ds.test_sets[tgt], pl.DataFrame)
            assert ds.test_sets[tgt].shape[0] == _CLASSIFY_INPUT_ROWS
            assert ds.test_sets[tgt].shape[1] == 56


# ---------------------------------------------------------------------------
# Step 7: concat
# ---------------------------------------------------------------------------

# Suite-mutation helper for step 7 — adds concat=ConcatDataSetSuite and the
# methods param list to the step 6 mutations.
_SUITE_METHODS = ("xgb", "dt")


def _inject_suite_concat_settings(config) -> None:
    """Mutate config to use ConcatDataSetSuite + ClassifyAllSuite + ModelSuite + methods."""
    config.data["step_class_set"]["steps"]["concat"] = "ConcatDataSetSuite"
    config.data["step_class_set"]["steps"]["classify"] = "ClassifyAllSuite"
    config.data["step_class_set"]["steps"]["model"] = "ModelSuite"
    config.data["step_param_set"]["steps"]["model"] = {"methods": ["XGB", "DT"]}


class TestClassifyConcatClassLoader:
    """Tests for load_classify_step7_concat_dataset (default + suite variants)."""

    def test_load_dataset_valid_config(self, classify_config_001):
        """Default config produces a ConcatDataSetAll with step_name='concat'."""
        ds = load_classify_step7_concat_dataset(classify_config_001)
        assert isinstance(ds, ConcatDataSetAll)
        assert ds.step_name == "concat"

    def test_load_suite_dataset_valid_config(self, classify_config_001):
        """Suite config produces a ConcatDataSetSuite with step_name='concat'."""
        _inject_suite_concat_settings(classify_config_001)
        ds = load_classify_step7_concat_dataset(classify_config_001)
        assert isinstance(ds, ConcatDataSetSuite)
        assert ds.step_name == "concat"

    def test_load_dataset_input_data(
        self, classify_config_001, test_data_file, training_dir,
    ):
        """Provided input_data + predictions populate the concat wrapper.

        Runs the full classify pipeline through step6 (read pre-trained models
        from tests/data/training/, run test_targets), then exercises step 7's
        loader on the resulting predictions.
        """
        pipeline = build_classify_prepare_pipeline(
            classify_config_001, test_data_file, stop_after="extract",
        )

        ds_classify = load_classify_step6_classify_dataset(
            classify_config_001, pipeline.extract.target_features,
        )
        # 3-target model file dict; the 2-target classify config only loads
        # temp + psal, but extra dict entries are ignored.
        ds_classify.model_file_names = {
            tgt: str(training_dir / f"model_{tgt}.joblib") for tgt in TARGETS_NONEMPTY
        }
        ds_classify.read_models()
        ds_classify.test_targets()

        ds = load_classify_step7_concat_dataset(
            classify_config_001, pipeline.input.input_data, ds_classify.predictions,
        )

        assert isinstance(ds, ConcatDataSetAll)
        for tgt in ("temp", "psal"):
            assert isinstance(ds.predictions[tgt], pl.DataFrame)
            assert ds.predictions[tgt].shape[0] == _CLASSIFY_INPUT_ROWS
            assert ds.predictions[tgt].shape[1] == 7

    def test_load_suite_dataset_input_data(
        self, classify_config_001, test_data_file, training_dir,
    ):
        """Same as test_load_dataset_input_data but for the suite variant.

        Suite predictions have a ``method`` column distinguishing per-method
        rows, so predictions["temp"] has 2× the input row count (2 methods).
        """
        _inject_suite_concat_settings(classify_config_001)
        pipeline = build_classify_prepare_pipeline(
            classify_config_001, test_data_file, stop_after="extract",
        )

        ds_classify = load_classify_step6_classify_dataset(
            classify_config_001, pipeline.extract.target_features,
        )
        # 4-key suite model dict (2 methods × 2 targets); 2-target config
        # ignores the pres entries.
        ds_classify.model_file_names = {
            f"{method}_{tgt}": str(training_dir / f"model_{tgt}_{method}.joblib")
            for method in _SUITE_METHODS
            for tgt in TARGETS_NONEMPTY
        }
        ds_classify.read_models()
        ds_classify.test_targets()

        ds = load_classify_step7_concat_dataset(
            classify_config_001, pipeline.input.input_data, ds_classify.predictions,
        )

        assert isinstance(ds, ConcatDataSetSuite)
        for tgt in ("temp", "psal"):
            assert isinstance(ds.predictions[tgt], pl.DataFrame)
            # 2 methods × input rows.
            assert ds.predictions[tgt].shape[0] == _CLASSIFY_INPUT_ROWS * 2
            # +1 column ('method') over the default ConcatDataSetAll predictions.
            assert ds.predictions[tgt].shape[1] == 8