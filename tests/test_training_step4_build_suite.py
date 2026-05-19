"""Unit tests for the ``BuildModelSuite`` class.

BuildModelSuite is the multi-model variant of BuildModel: it uses a
``ModelSuite`` as its base model and runs *several* methods in parallel
(here just XGB and DT, to keep tests fast). Composite keys are used
throughout — ``xgb_temp``, ``dt_temp``, etc. — instead of just ``temp``,
because each target has multiple trained models.

Refactored from the original which:
- Used a module-level ``setup_training_step4(test_obj)`` helper to mutate the
  config to suite settings and load the training input (replaced here by the
  ``training_config_001_suite`` fixture for the mutation + the conftest
  ``training_input_001`` fixture for the input wiring)
- Had ~30 lines of per-target × per-output-kind triplication in
  ``test_write_aggregated_results`` (collapsed via a nested loop over
  TARGETS and output kinds)
- Triplicated the model-file path setup in ``test_write_models``
- All other tests followed the per-target triplication pattern (replaced
  with ``for tgt in TARGETS`` loops)

No 9-model fan-out here: BuildModelSuite tests the *combined* behaviour of
multiple methods running together, not each model wrapper individually.
"""

import os

import matplotlib
import polars as pl
import pytest

# Use non-interactive backend so plot tests don't try to open windows.
matplotlib.use("Agg")

from aiqclib.train.models.model_suite import ModelSuite
from aiqclib.train.step4_build_model.build_model_suite import BuildModelSuite

from tests.conftest import TARGETS_NONEMPTY


# Composite keys produced by ModelSuite when methods=["XGB", "DT"]:
# one model per (method, target) combination, 2 × 3 = 6 keys.
SUITE_METHODS = ("xgb", "dt")
SUITE_KEYS = tuple(
    f"{method}_{tgt}" for method in SUITE_METHODS for tgt in TARGETS_NONEMPTY
)


# ---------------------------------------------------------------------------
# Suite-config fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def training_config_001_suite(training_config_001_bo002):
    """training_config_001 with suite settings injected.

    Three mutations applied to a fresh training_config_001:
    - ``step_class_set.steps.build`` = "BuildModelSuite"
    - ``step_class_set.steps.model`` = "ModelSuite"
    - ``step_param_set.steps.model.methods`` = ["XGB", "DT"]

    Only two methods are tested (not all 9 defaults) to keep tests fast.
    """
    training_config_001_bo002.data["step_class_set"]["steps"]["build"] = (
        "BuildModelSuite"
    )
    training_config_001_bo002.data["step_class_set"]["steps"]["model"] = "ModelSuite"
    training_config_001_bo002.data["step_param_set"]["steps"]["model"] = {
        "methods": ["XGB", "DT"],
    }
    return training_config_001_bo002


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBuildModelSuite:
    """Tests for BuildModelSuite's multi-model build, test, save behaviour."""

    # ----- Identity / config -----

    def test_step_name(self, training_config_001_suite):
        """step_name == 'build'."""
        ds = BuildModelSuite(training_config_001_suite)
        assert ds.step_name == "build"

    def test_multi_flag_check(self, training_config_001_suite):
        """Constructing BuildModelSuite with a single-model wrapper raises ValueError.

        The error message must mention ``multi=True`` because that's the
        invariant the user has to fix.
        """
        training_config_001_suite.data["step_class_set"]["steps"]["model"] = "XGBoost"
        with pytest.raises(ValueError, match="multi=True"):
            _ = BuildModelSuite(training_config_001_suite)

    def test_shap_flag(self, training_config_001_suite):
        """``calculate_shap`` propagates to both the suite and each child method.

        The suite's own ``enable_shap`` and every ``method_obj.enable_shap``
        must move together. BuildModelSuite, like BuildModel, honours the
        flag (SHAP is computed at the testing stage).
        """
        # Unset == False, both at suite level and per-method.
        ds = BuildModelSuite(training_config_001_suite)
        assert ds.base_model.enable_shap is False
        for method_obj in ds.base_model.method_objs.values():
            assert method_obj.enable_shap is False

        # True at suite level propagates to every method.
        training_config_001_suite.data["step_param_set"]["steps"]["model"][
            "calculate_shap"
        ] = True
        ds = BuildModelSuite(training_config_001_suite)
        assert ds.base_model.enable_shap is True
        for method_obj in ds.base_model.method_objs.values():
            assert method_obj.enable_shap is True

        # False at suite level also propagates.
        training_config_001_suite.data["step_param_set"]["steps"]["model"][
            "calculate_shap"
        ] = False
        ds = BuildModelSuite(training_config_001_suite)
        assert ds.base_model.enable_shap is False
        for method_obj in ds.base_model.method_objs.values():
            assert method_obj.enable_shap is False

    def test_output_file_names(self, training_config_001_suite):
        """Model files use composite ``{method}_{tgt}`` keys; outputs use ``{tgt}``."""
        ds = BuildModelSuite(training_config_001_suite)
        model_base = "/path/to/model_1/nrt_bo_001/model_folder_1"
        out_base = "/path/to/build_1/nrt_bo_001/build_folder_1"

        # Model files: one per (method, target) composite key
        for method in SUITE_METHODS:
            for tgt in TARGETS_NONEMPTY:
                key = f"{method}_{tgt}"
                assert (
                    str(ds.model_file_names[key]) == f"{model_base}/model_{key}.joblib"
                )

        # Aggregated output files: one per target (not per method)
        for tgt in TARGETS_NONEMPTY:
            assert (
                str(ds.output_file_names["report"][tgt])
                == f"{out_base}/test_report_{tgt}.tsv"
            )
            assert (
                str(ds.output_file_names["contingency_table"][tgt])
                == f"{out_base}/test_contingency_tables_{tgt}.parquet"
            )
            assert (
                str(ds.output_file_names["shap_value"][tgt])
                == f"{out_base}/test_shap_values_{tgt}.parquet"
            )
            assert (
                str(ds.output_file_names["metric_plot"][tgt])
                == f"{out_base}/test_metric_plots_{tgt}.svg"
            )

    def test_base_model(self, training_config_001_suite):
        """The base_model is a ModelSuite instance."""
        ds = BuildModelSuite(training_config_001_suite)
        assert isinstance(ds.base_model, ModelSuite)

    # ----- Build / test pipeline behaviour -----

    def test_build_final_model_targets(
        self, training_config_001_suite, training_input_001
    ):
        """build_final_model_targets populates final_models with all composite keys.

        Each (method, target) combination produces a distinct model, and the
        training_set passed to each model spans both train + test data
        (because the "final" model uses everything).
        """
        ds = BuildModelSuite(
            training_config_001_suite,
            training_sets=training_input_001.training_sets,
            test_sets=training_input_001.test_sets,
        )
        ds.build_final_model_targets()

        # All composite keys exist
        for key in SUITE_KEYS:
            assert key in ds.final_models

        # XGB and DT produce distinct objects for the same target
        for tgt in TARGETS_NONEMPTY:
            assert ds.final_models[f"xgb_{tgt}"] is not ds.final_models[f"dt_{tgt}"]

        # final_models combine train + test data. Temp had 22 train + 2 test rows.
        assert ds.final_models["xgb_temp"].training_set.height == 22 + 2

    def test_build_targets(self, training_config_001_suite, training_input_001):
        """build_targets populates models with composite keys; sees only training data."""
        ds = BuildModelSuite(
            training_config_001_suite,
            training_sets=training_input_001.training_sets,
            test_sets=training_input_001.test_sets,
        )
        ds.build_targets()

        for key in SUITE_KEYS:
            assert key in ds.models
        for tgt in TARGETS_NONEMPTY:
            assert ds.models[f"xgb_{tgt}"] is not ds.models[f"dt_{tgt}"]

        # models use train-only data
        assert ds.models["xgb_temp"].training_set.height == 22

    def test_test_targets(self, training_config_001_suite, training_input_001):
        """test_targets aggregates per-method predictions into a 'method' column.

        With 2 methods, each target's prediction count doubles vs. single-model.
        """
        ds = BuildModelSuite(
            training_config_001_suite,
            training_sets=training_input_001.training_sets,
            test_sets=training_input_001.test_sets,
        )
        ds.build_targets()
        ds.test_targets()

        # Aggregated predictions: include 'method' column, 2x rows per target
        # Temp: 2 test rows × 2 methods = 4
        # Psal: 2 test rows × 2 methods = 4
        # Pres: 0 test rows × 2 methods = 0
        expected_pred_heights = {"temp": 4, "psal": 4, "pres": 0}
        for tgt in TARGETS_NONEMPTY:
            assert isinstance(ds.predictions[tgt], pl.DataFrame)
            assert "method" in ds.predictions[tgt].columns
            assert ds.predictions[tgt].shape[0] == expected_pred_heights[tgt]

        # Contingency tables aggregate the same way
        for tgt in TARGETS_NONEMPTY:
            assert isinstance(ds.contingency_tables[tgt], pl.DataFrame)
            assert "method" in ds.contingency_tables[tgt].columns
            assert ds.contingency_tables[tgt].height == expected_pred_heights[tgt]

        # Reports include the 'method' column
        for tgt in TARGETS_NONEMPTY:
            assert isinstance(ds.reports[tgt], pl.DataFrame)
            assert "method" in ds.reports[tgt].columns

    # ----- Error cases (no data / wrong order of calls) -----

    def test_build_without_data(self, training_config_001_suite):
        """build_targets with no inputs raises ValueError."""
        ds = BuildModelSuite(
            training_config_001_suite, training_sets=None, test_sets=None
        )
        with pytest.raises(ValueError):
            ds.build_targets()

    def test_build_final_model_without_test_data(
        self, training_config_001_suite, training_input_001
    ):
        """build_final_model_targets with test_sets=None raises ValueError."""
        ds = BuildModelSuite(
            training_config_001_suite,
            training_sets=training_input_001.training_sets,
            test_sets=None,
        )
        with pytest.raises(ValueError):
            ds.build_final_model_targets()

    def test_build_final_model_without_training_data(self, training_config_001_suite):
        """build_final_model_targets with no inputs raises ValueError."""
        ds = BuildModelSuite(
            training_config_001_suite, training_sets=None, test_sets=None
        )
        with pytest.raises(ValueError):
            ds.build_final_model_targets()

    def test_test_without_model(self, training_config_001_suite, training_input_001):
        """test_targets before build_targets raises ValueError."""
        ds = BuildModelSuite(
            training_config_001_suite,
            training_sets=training_input_001.training_sets,
            test_sets=training_input_001.test_sets,
        )
        with pytest.raises(ValueError):
            ds.test_targets()

    def test_empty_write_calls(self, training_config_001_suite, training_input_001):
        """Calling any write_*/create_metric_plots before producing data raises ValueError.

        Five sibling checks for the five write paths. Original had this
        as a single test with five with-statements; preserved structure but
        slightly compacted.
        """
        ds = BuildModelSuite(
            training_config_001_suite,
            training_sets=training_input_001.training_sets,
            test_sets=training_input_001.test_sets,
        )
        with pytest.raises(ValueError):
            ds.write_reports()
        with pytest.raises(ValueError):
            ds.write_contingency_tables()
        with pytest.raises(ValueError):
            ds.create_metric_plots()
        with pytest.raises(ValueError):
            ds.write_predictions()
        with pytest.raises(ValueError):
            ds.write_models()

    def test_read_models_no_file(
        self, training_config_001_suite, training_input_001, training_dir
    ):
        """Missing model files raise FileNotFoundError."""
        ds = BuildModelSuite(
            training_config_001_suite,
            training_sets=None,
            test_sets=training_input_001.test_sets,
        )
        ds.model_file_names["xgb_temp"] = str(
            training_dir / "non_existent_model.joblib"
        )

        with pytest.raises(FileNotFoundError):
            ds.read_models()

    # ----- File output -----

    def test_write_aggregated_results(
        self, training_config_001_suite, training_input_001, test_output_dir
    ):
        """All five aggregated output kinds write per-target files.

        The original wrote 15 paths (5 kinds × 3 targets) by hand, then
        asserted existence for all 15, then removed all 15 — ~90 lines.
        Same coverage here, ~15 lines via nested loops.
        """
        training_config_001_suite.data["step_param_set"]["steps"]["model"][
            "calculate_shap"
        ] = True
        ds = BuildModelSuite(
            training_config_001_suite,
            training_sets=training_input_001.training_sets,
            test_sets=training_input_001.test_sets,
        )

        # (output_kind, filename_template) pairs. {tgt} substituted per target.
        output_specs = [
            ("report", "test_test_report_{tgt}.tsv"),
            ("contingency_table", "test_test_contingency_{tgt}.parquet"),
            ("shap_value", "test_test_shap_values_{tgt}.parquet"),
            ("prediction", "test_test_prediction_{tgt}.parquet"),
            ("metric_plot", "test_test_metric_plot_{tgt}.svg"),
        ]

        # Wire all 15 output paths
        output_paths: dict[str, dict[str, str]] = {}
        for kind, template in output_specs:
            output_paths[kind] = {
                tgt: str(test_output_dir / template.format(tgt=tgt))
                for tgt in TARGETS_NONEMPTY
            }
            ds.output_file_names[kind] = output_paths[kind]

        ds.build_targets()
        ds.test_targets()

        ds.write_reports()
        ds.write_contingency_tables()
        ds.write_shap_values()
        ds.write_predictions()
        ds.create_metric_plots()

        for kind, _ in output_specs:
            for tgt in TARGETS_NONEMPTY:
                assert os.path.exists(output_paths[kind][tgt])
                os.remove(output_paths[kind][tgt])  # comment out to debug

    def test_write_models(
        self, training_config_001_suite, training_input_001, test_output_dir
    ):
        """write_models serialises one joblib per composite key (6 files for XGB+DT × 3 targets)."""
        ds = BuildModelSuite(
            training_config_001_suite,
            training_sets=training_input_001.training_sets,
            test_sets=training_input_001.test_sets,
        )

        output_paths = {
            key: str(test_output_dir / f"test_model_{key}.joblib") for key in SUITE_KEYS
        }
        ds.model_file_names = output_paths

        ds.build_final_model_targets()
        ds.write_models()

        for key in SUITE_KEYS:
            assert os.path.exists(output_paths[key])
            os.remove(output_paths[key])  # comment out to debug
