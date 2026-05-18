"""Unit tests for the ``KFoldValidationSuite`` class.

KFoldValidationSuite is the multi-model variant of KFoldValidation: it runs
k-fold cross-validation across several methods in parallel (here just XGB
and DT, to keep tests fast). Composite keys are used throughout —
``xgb_temp``, ``dt_psal``, etc. — instead of just ``temp``, because each
target has multiple models.

Refactored from a single ``unittest.TestCase`` class with a module-level
``setup_training_step2`` helper. The helper is replaced by the
``training_input_001`` fixture from conftest combined with a file-local
``training_config_001_validate_suite`` fixture that injects the
KFoldValidationSuite + ModelSuite + methods=[XGB, DT] mutations. The three
file-output tests had ~25 lines of per-key hand-written paths each;
collapsed into nested loops over ``SUITE_KEYS``.

Note on SHAP behaviour:
Unlike BuildModelSuite (step4), which propagates ``calculate_shap`` to the
underlying ModelSuite (SHAP is computed at the testing stage),
KFoldValidationSuite explicitly **does not** propagate the flag — validation
never uses SHAP, regardless of config. ``test_shap_flag`` defends this
distinction.
"""

import os

import matplotlib
import polars as pl
import pytest

# Non-interactive backend so plot tests don't open windows.
matplotlib.use("Agg")

from aiqclib.train.models.model_suite import ModelSuite
from aiqclib.train.step2_validate_model.kfold_validation_suite import (
    KFoldValidationSuite,
)

from tests.conftest import TARGETS


# Composite keys produced by ModelSuite when methods=["XGB", "DT"]:
# 2 methods × 3 targets = 6 keys.
SUITE_METHODS = ("xgb", "dt")
SUITE_KEYS = tuple(f"{method}_{tgt}" for method in SUITE_METHODS for tgt in TARGETS)


# ---------------------------------------------------------------------------
# Suite-config fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def training_config_001_validate_suite(training_config_001):
    """training_config_001 with KFoldValidationSuite + ModelSuite + 2 methods injected.

    Three mutations applied to a fresh training_config_001:
    - ``step_class_set.steps.validate`` = "KFoldValidationSuite"
    - ``step_class_set.steps.model``    = "ModelSuite"
    - ``step_param_set.steps.model``    = {"methods": ["XGB", "DT"]}

    Only two methods are tested (not all 9 defaults) to keep tests fast.
    Because pytest shares the underlying ``training_config_001`` instance
    across fixtures within a test, the conftest ``training_input_001``
    fixture (which depends on ``training_config_001``) will see these
    mutations — but the input loading only uses the target_set, not the
    model class, so process_targets still produces 3-target training data.
    """
    training_config_001.data["step_class_set"]["steps"]["validate"] = (
        "KFoldValidationSuite"
    )
    training_config_001.data["step_class_set"]["steps"]["model"] = "ModelSuite"
    training_config_001.data["step_param_set"]["steps"]["model"] = {
        "methods": ["XGB", "DT"],
    }
    return training_config_001


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestKFoldValidationSuite:
    """Tests for KFoldValidationSuite's multi-method k-fold + file output."""

    # ----- Identity / config -----

    def test_step_name(self, training_config_001_validate_suite):
        """step_name == 'validate'."""
        ds = KFoldValidationSuite(training_config_001_validate_suite)
        assert ds.step_name == "validate"

    def test_multi_flag_check(self, training_config_001_validate_suite):
        """Constructing KFoldValidationSuite with a single-model wrapper raises ValueError.

        The error message must mention ``multi=True`` because that's the
        invariant the user has to fix.
        """
        training_config_001_validate_suite.data["step_class_set"]["steps"]["model"] = (
            "XGBoost"
        )
        with pytest.raises(ValueError, match="multi=True"):
            _ = KFoldValidationSuite(training_config_001_validate_suite)

    def test_shap_flag(self, training_config_001_validate_suite):
        """``calculate_shap`` is NOT propagated by KFoldValidationSuite.

        Validation never computes SHAP regardless of config. The suite's
        ``enable_shap`` stays False even when the config sets
        ``calculate_shap=True``, and the per-method ``enable_shap`` does
        too. Contrast with BuildModelSuite, which propagates the flag.
        """
        # Unset == False (suite and per-method).
        ds = KFoldValidationSuite(training_config_001_validate_suite)
        assert ds.base_model.enable_shap is False
        for method_obj in ds.base_model.method_objs.values():
            assert method_obj.enable_shap is False

        # Setting True at config level does NOT propagate for validation.
        training_config_001_validate_suite.data["step_param_set"]["steps"]["model"][
            "calculate_shap"
        ] = True
        ds = KFoldValidationSuite(training_config_001_validate_suite)
        assert ds.base_model.enable_shap is False
        for method_obj in ds.base_model.method_objs.values():
            assert method_obj.enable_shap is False

        # Explicit False also stays False.
        training_config_001_validate_suite.data["step_param_set"]["steps"]["model"][
            "calculate_shap"
        ] = False
        ds = KFoldValidationSuite(training_config_001_validate_suite)
        assert ds.base_model.enable_shap is False
        for method_obj in ds.base_model.method_objs.values():
            assert method_obj.enable_shap is False

    def test_output_file_names_init(self, training_config_001_validate_suite):
        """Default output paths retain the ``{method}`` placeholder until process_targets runs.

        Before process_targets dynamically rewrites the paths with concrete
        method names, the per-target paths contain the unrendered ``{method}``
        substring. Original checked only ``temp``; preserving that.
        """
        ds = KFoldValidationSuite(training_config_001_validate_suite)
        base = "/path/to/validate_1/nrt_bo_001/validate_folder_1"

        assert (
            str(ds.output_file_names["report"]["temp"])
            == f"{base}/validation_report_{{method}}_temp.tsv"
        )
        assert (
            str(ds.output_file_names["contingency_table"]["temp"])
            == f"{base}/contingency_tables_{{method}}_temp.parquet"
        )
        assert (
            str(ds.output_file_names["metric_plot"]["temp"])
            == f"{base}/metric_plots_{{method}}_temp.svg"
        )

    def test_base_model(self, training_config_001_validate_suite):
        """base_model is a ModelSuite instance."""
        ds = KFoldValidationSuite(training_config_001_validate_suite)
        assert isinstance(ds.base_model, ModelSuite)

    # ----- Validation behaviour -----

    def test_fold_validation(self, training_config_001_validate_suite, training_input_001):
        """process_targets populates reports + contingency_tables for every composite key.

        After process_targets, all six SUITE_KEYS exist in both .reports and
        .contingency_tables. Dynamic path rewriting has also substituted the
        ``{method}`` placeholder with the actual method name.
        """
        ds = KFoldValidationSuite(
            training_config_001_validate_suite,
            training_sets=training_input_001.training_sets,
        )
        ds.process_targets()

        # All composite keys exist in both reports and contingency tables.
        for key in SUITE_KEYS:
            assert key in ds.reports
            assert key in ds.contingency_tables

        # Spot-check one report's shape. k=3 folds × 6 metric rows = 18; cols structural.
        assert isinstance(ds.reports["xgb_temp"], pl.DataFrame)
        assert ds.reports["xgb_temp"].shape[0] == 18
        assert ds.reports["xgb_temp"].shape[1] == 8

        # Spot-check one contingency table's height (sum of all validation-fold rows).
        assert isinstance(ds.contingency_tables["dt_psal"], pl.DataFrame)
        assert ds.contingency_tables["dt_psal"].height == 34
        assert ds.contingency_tables["dt_psal"].columns == [
            "k", "label", "predicted_label", "score",
        ]

        # The ``{method}`` placeholder has been substituted with concrete names.
        assert (
            str(ds.output_file_names["report"]["xgb_temp"])
            == "/path/to/validate_1/nrt_bo_001/validate_folder_1/validation_report_xgb_temp.tsv"
        )

    # ----- File output (per-key write, assert exists, manually remove) -----

    def test_write_reports(
        self, training_config_001_validate_suite, training_input_001, test_output_dir,
    ):
        """write_reports produces a TSV per composite key (6 files: XGB+DT × 3 targets)."""
        ds = KFoldValidationSuite(
            training_config_001_validate_suite,
            training_sets=training_input_001.training_sets,
        )
        ds.process_targets()

        output_paths = {
            key: str(test_output_dir / f"test_validation_report_{key}.tsv")
            for key in SUITE_KEYS
        }
        for key in SUITE_KEYS:
            ds.output_file_names["report"][key] = output_paths[key]

        ds.write_reports()

        for key in SUITE_KEYS:
            assert os.path.exists(output_paths[key])
            os.remove(output_paths[key])  # comment out to debug

    def test_write_contingency_tables(
        self, training_config_001_validate_suite, training_input_001, test_output_dir,
    ):
        """write_contingency_tables produces a parquet per composite key."""
        ds = KFoldValidationSuite(
            training_config_001_validate_suite,
            training_sets=training_input_001.training_sets,
        )
        ds.process_targets()

        output_paths = {
            key: str(test_output_dir / f"test_contingency_{key}.parquet")
            for key in SUITE_KEYS
        }
        for key in SUITE_KEYS:
            ds.output_file_names["contingency_table"][key] = output_paths[key]

        ds.write_contingency_tables()

        for key in SUITE_KEYS:
            assert os.path.exists(output_paths[key])
            os.remove(output_paths[key])  # comment out to debug

    def test_create_metric_plots(
        self, training_config_001_validate_suite, training_input_001, test_output_dir,
    ):
        """create_metric_plots produces an SVG per composite key."""
        ds = KFoldValidationSuite(
            training_config_001_validate_suite,
            training_sets=training_input_001.training_sets,
        )
        ds.process_targets()

        output_paths = {
            key: str(test_output_dir / f"test_metric_plots_{key}.svg")
            for key in SUITE_KEYS
        }
        for key in SUITE_KEYS:
            ds.output_file_names["metric_plot"][key] = output_paths[key]

        ds.create_metric_plots()

        for key in SUITE_KEYS:
            assert os.path.exists(output_paths[key])
            os.remove(output_paths[key])  # comment out to debug

    # ----- Error cases (write before process_targets) -----

    def test_write_reports_empty_reports(
        self, training_config_001_validate_suite, training_input_001,
    ):
        """write_reports before process_targets raises ValueError."""
        ds = KFoldValidationSuite(
            training_config_001_validate_suite,
            training_sets=training_input_001.training_sets,
        )
        with pytest.raises(ValueError):
            ds.write_reports()

    def test_write_contingency_tables_empty(
        self, training_config_001_validate_suite, training_input_001,
    ):
        """write_contingency_tables before process_targets raises ValueError."""
        ds = KFoldValidationSuite(
            training_config_001_validate_suite,
            training_sets=training_input_001.training_sets,
        )
        with pytest.raises(ValueError):
            ds.write_contingency_tables()

    def test_create_metric_plots_empty(
        self, training_config_001_validate_suite, training_input_001,
    ):
        """create_metric_plots before process_targets raises ValueError."""
        ds = KFoldValidationSuite(
            training_config_001_validate_suite,
            training_sets=training_input_001.training_sets,
        )
        with pytest.raises(ValueError):
            ds.create_metric_plots()