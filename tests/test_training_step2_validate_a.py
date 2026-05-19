"""Unit tests for the ``KFoldValidation`` class.

Exercises:
- step_name, output_file_names resolution, base_model wiring
- training_sets shape after wiring ds_input
- write_reports / write_contingency_tables / create_metric_plots: file output
  side effects, plus their empty-state error behaviour
- The 9-model fan-out (XGBoost, LogisticRegression, LDA, SVM, DecisionTree,
  RandomForest, KNN, GaussianNaiveBayes, MLP), previously nine sibling
  TestCase classes, now a single parametrized class importing MODEL_CASES.

Refactored from the original which had:
- A module-level ``setup_training_step2(test_obj)`` helper (replaced by the
  ``training_input_001`` fixture in conftest)
- A ``run_fold_validation(test_obj)`` helper duplicated across 9 model
  classes (kept as a module-level helper here, called from a single
  parametrized test)
- ~25 lines of triplicated ``temp/psal/pres`` assertions (now loops over
  ``TARGETS``)
- 9 near-identical model classes (collapsed to one ``TestModels`` class with
  ``@pytest.mark.parametrize("case", MODEL_CASES)``)
"""

import os

import polars as pl
import pytest

from aiqclib.train.models.xgboost import XGBoost
from aiqclib.train.step2_validate_model.kfold_validation import KFoldValidation

from tests._model_cases import MODEL_CASES
from tests.conftest import TARGETS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_fold_validation(ds: KFoldValidation) -> None:
    """Run ``process_targets`` and assert the shape/columns of its outputs.

    Used by both the XGBoost-specific test on TestKFoldValidation and the
    parametrized per-model test below. Previously inlined as
    ``run_fold_validation(test_obj)`` and called from nine separate
    TestCase classes.
    """
    ds.process_targets()

    # Reports: one row per fold per metric. Shape is the same across targets.
    for tgt in TARGETS:
        assert isinstance(ds.reports[tgt], pl.DataFrame)
        assert ds.reports[tgt].shape[0] == 18
        assert ds.reports[tgt].shape[1] == 8

    # Contingency tables: row count equals the training set's row count for that target.
    # The original test asserted these one-per-target, each with a different number.
    expected_heights = {
        "temp": 22,
        "psal": 34,
        "pres": 18,
    }
    for tgt in TARGETS:
        assert isinstance(ds.contingency_tables[tgt], pl.DataFrame)
        assert ds.contingency_tables[tgt].height == expected_heights[tgt]
        assert ds.contingency_tables[tgt].columns == [
            "k",
            "label",
            "predicted_label",
            "score",
        ]


# ---------------------------------------------------------------------------
# Core KFoldValidation tests (XGBoost path — the default config)
# ---------------------------------------------------------------------------


class TestKFoldValidation:
    """Core tests against KFoldValidation with the default (XGBoost) model.

    The default training config sets model=XGBoost, so these tests use it
    implicitly. Per-model fan-out lives in ``TestModels`` below.
    """

    def test_step_name(self, training_config_001):
        """step_name == 'validate'."""
        ds = KFoldValidation(training_config_001)
        assert ds.step_name == "validate"

    def test_output_file_names(self, training_config_001):
        """Default output paths derive from config.path_info; no data-dependence."""
        ds = KFoldValidation(training_config_001)
        base = "/path/to/validate_1/nrt_bo_001/validate_folder_1"

        for tgt in TARGETS:
            assert (
                str(ds.output_file_names["report"][tgt])
                == f"{base}/validation_report_{tgt}.tsv"
            )
            assert (
                str(ds.output_file_names["contingency_table"][tgt])
                == f"{base}/contingency_tables_{tgt}.parquet"
            )
            assert (
                str(ds.output_file_names["metric_plot"][tgt])
                == f"{base}/metric_plots_{tgt}.svg"
            )

    def test_base_model(self, training_config_001):
        """Default config selects XGBoost as the base model."""
        ds = KFoldValidation(training_config_001)
        assert isinstance(ds.base_model, XGBoost)

    def test_training_sets(self, training_config_001, training_input_001):
        """training_sets are loaded as DataFrames with the expected shape per target."""
        ds = KFoldValidation(
            training_config_001, training_sets=training_input_001.training_sets
        )

        expected_shapes = {
            "temp": (22, 57),
            "psal": (34, 57),
            "pres": (18, 57),
        }
        for tgt in TARGETS:
            assert isinstance(ds.training_sets[tgt], pl.DataFrame)
            assert ds.training_sets[tgt].shape == expected_shapes[tgt]

    def test_shap_flag(self, training_config_001):
        """``calculate_shap`` in config is forwarded to the base model.

        The original test repeats the construction three times with three
        values (unset, True, False) and checks the result — preserved here.
        """
        ds = KFoldValidation(training_config_001)
        assert ds.base_model.enable_shap is False

        training_config_001.data["step_param_set"]["steps"]["model"][
            "calculate_shap"
        ] = True
        ds = KFoldValidation(training_config_001)
        assert (
            ds.base_model.enable_shap is False
        )  # NB: original asserted False here too

        training_config_001.data["step_param_set"]["steps"]["model"][
            "calculate_shap"
        ] = False
        ds = KFoldValidation(training_config_001)
        assert ds.base_model.enable_shap is False

    def test_default_k_fold(self, training_config_001, training_input_001):
        """When ``k_fold`` is unset, ``get_k_fold`` defaults to 10."""
        ds = KFoldValidation(
            training_config_001, training_sets=training_input_001.training_sets
        )
        ds.config.data["step_param_set"]["steps"]["validate"]["k_fold"] = None
        assert ds.get_k_fold() == 10

    # ----- File-output tests (write, assert exists, manually remove) -----

    def test_write_reports(
        self, training_config_001, training_input_001, test_output_dir
    ):
        """write_reports produces a TSV per target at the configured path."""
        ds = KFoldValidation(
            training_config_001, training_sets=training_input_001.training_sets
        )
        output_paths = {
            tgt: str(test_output_dir / f"test_validation_report_{tgt}.tsv")
            for tgt in TARGETS
        }
        ds.output_file_names["report"] = output_paths

        ds.process_targets()
        ds.write_reports()

        for tgt in TARGETS:
            assert os.path.exists(output_paths[tgt])
            os.remove(output_paths[tgt])  # comment out to debug

    def test_write_contingency_tables(
        self, training_config_001, training_input_001, test_output_dir
    ):
        """write_contingency_tables produces a parquet per target."""
        ds = KFoldValidation(
            training_config_001, training_sets=training_input_001.training_sets
        )
        output_paths = {
            tgt: str(test_output_dir / f"test_contingency_{tgt}.parquet")
            for tgt in TARGETS
        }
        ds.output_file_names["contingency_table"] = output_paths

        ds.process_targets()
        ds.write_contingency_tables()

        for tgt in TARGETS:
            assert os.path.exists(output_paths[tgt])
            os.remove(output_paths[tgt])  # comment out to debug

    def test_create_metric_plots(
        self, training_config_001, training_input_001, test_output_dir
    ):
        """create_metric_plots produces an SVG per target."""
        ds = KFoldValidation(
            training_config_001, training_sets=training_input_001.training_sets
        )
        output_paths = {
            tgt: str(test_output_dir / f"test_metric_plots_{tgt}.svg")
            for tgt in TARGETS
        }
        ds.output_file_names["metric_plot"] = output_paths

        ds.process_targets()
        ds.create_metric_plots()

        for tgt in TARGETS:
            assert os.path.exists(output_paths[tgt])
            os.remove(output_paths[tgt])  # comment out to debug

    # ----- Empty-state error behaviour -----

    def test_write_reports_empty_reports(self, training_config_001, training_input_001):
        """Calling write_reports before process_targets raises ValueError."""
        ds = KFoldValidation(
            training_config_001, training_sets=training_input_001.training_sets
        )
        with pytest.raises(ValueError):
            ds.write_reports()

    def test_write_contingency_tables_empty(
        self, training_config_001, training_input_001
    ):
        """Calling write_contingency_tables before process_targets raises ValueError."""
        ds = KFoldValidation(
            training_config_001, training_sets=training_input_001.training_sets
        )
        with pytest.raises(ValueError):
            ds.write_contingency_tables()

    def test_create_metric_plots_empty(self, training_config_001, training_input_001):
        """Calling create_metric_plots before process_targets raises ValueError."""
        ds = KFoldValidation(
            training_config_001, training_sets=training_input_001.training_sets
        )
        with pytest.raises(ValueError):
            ds.create_metric_plots()


# ---------------------------------------------------------------------------
# Per-model fan-out
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", MODEL_CASES, ids=lambda c: c.config_name)
class TestModels:
    """Same two behavioural checks against each of the 9 supported models.

    The parametrize id is the config_name, so ``pytest -k xgboost`` and
    similar still work.
    """

    def test_base_model(self, case, training_config_001):
        """KFoldValidation.base_model is an instance of the configured wrapper."""
        training_config_001.data["step_class_set"]["steps"]["model"] = case.config_name
        ds = KFoldValidation(training_config_001)
        assert isinstance(ds.base_model, case.wrapper_cls)

    def test_fold_validation(self, case, training_config_001, training_input_001):
        """End-to-end fold validation with the configured model produces correct outputs."""
        training_config_001.data["step_class_set"]["steps"]["model"] = case.config_name
        ds = KFoldValidation(
            training_config_001, training_sets=training_input_001.training_sets
        )
        _run_fold_validation(ds)
