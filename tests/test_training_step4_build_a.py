"""Unit tests for the ``BuildModel`` class.

Exercises:
- Identity and config wiring (step_name, output_file_names, training_sets shape)
- The ``calculate_shap`` flag: unlike KFoldValidation (validation never uses
  SHAP), BuildModel honours the flag because SHAP is computed at the testing
  stage. ``test_shap_flag`` defends that invariant.
- Default-config XGBoost path: ``test_train_with_xgboost``,
  ``test_train_final_model_with_xgboost``, ``test_read_models`` — these
  verify the default config (model=XGBoost) produces XGBoost instances and
  that on-disk model files load correctly.
- Distinct-object invariants: ``test_model_objects``,
  ``test_test_model_objects`` (each target gets its own model instance, not
  a shared reference).
- File-output behaviour: write_reports, write_contingency_tables,
  write_shap_values, write_models, write_predictions, create_metric_plots,
  plus their empty-state ValueError counterparts.
- The 9-model fan-out (XGBoost, LogisticRegression, LDA, SVM, DecisionTree,
  RandomForest, KNN, GaussianNaiveBayes, MLP), previously nine sibling
  TestCase classes, now a single parametrized ``TestModels`` class.

Refactored from the original which had:
- A module-level ``setup_training_step4(test_obj)`` helper (replaced by the
  ``training_input_001`` fixture in conftest)
- A ``run_test_with_trained_model(test_obj)`` helper called from 9 per-model
  classes (kept as a module-level helper here, called from a single
  parametrized test)
- ~30 lines of per-target triplication per file-output test (now loops over
  ``TARGETS``)
- Nine ``Test{Model}`` classes (collapsed into ``TestModels`` parametrized
  by MODEL_CASES)
- A latent **bug** in the original: each per-model class had a
  ``test_base_model`` that checked ``KFoldValidation`` (step2's class) rather
  than ``BuildModel`` (step4's class) — copy-paste from step2. The refactor
  fixes this to check ``BuildModel``, which is what the test name claims.
"""

import os

import polars as pl
import pytest

from aiqclib.train.models.xgboost import XGBoost
from aiqclib.train.step4_build_model.build_model import BuildModel

from tests._model_cases import MODEL_CASES
from tests.conftest import TARGETS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_test_with_trained_model(ds: BuildModel) -> None:
    """Build, test, then assert the shape/columns of the test sets and tables.

    Used by the parametrized ``test_model_output`` below. Previously inlined
    as ``run_test_with_trained_model(test_obj)`` and called from nine separate
    TestCase classes.
    """
    ds.build_targets()
    ds.test_targets()

    # Test sets / predictions: per-target shapes.
    expected_test_shapes = {
        "temp": (12, 56),  # TODO: update to actual value after data reduction
        "psal": (14, 56),  # TODO: update to actual value after data reduction
        "pres": (12, 56),  # TODO: update to actual value after data reduction
    }
    for tgt in TARGETS:
        assert isinstance(ds.test_sets[tgt], pl.DataFrame)
        assert ds.test_sets[tgt].shape == expected_test_shapes[tgt]

    # Contingency tables: height matches the test set rows; columns fixed.
    expected_heights = {
        "temp": 12,  # TODO: update to actual value after data reduction
        "psal": 14,  # TODO: update to actual value after data reduction
        "pres": 12,  # TODO: update to actual value after data reduction
    }
    for tgt in TARGETS:
        assert isinstance(ds.contingency_tables[tgt], pl.DataFrame)
        assert ds.contingency_tables[tgt].height == expected_heights[tgt]
        assert ds.contingency_tables[tgt].columns == [
            "k", "label", "predicted_label", "score",
        ]


# ---------------------------------------------------------------------------
# Core BuildModel tests (default config — XGBoost path)
# ---------------------------------------------------------------------------

class TestBuildModel:
    """Core tests against BuildModel with the default (XGBoost) model.

    Per-model fan-out lives in ``TestModels`` below. Tests here verify either
    behaviour independent of the model choice (config wiring, error cases,
    distinct-instance invariants) or the default-XGBoost path specifically
    (test_train_with_xgboost, test_read_models).
    """

    # ----- Identity / config -----

    def test_step_name(self, training_config_001):
        """step_name == 'build'."""
        ds = BuildModel(training_config_001)
        assert ds.step_name == "build"

    def test_output_file_names(self, training_config_001):
        """Default output paths derive from config.path_info; no data-dependence."""
        ds = BuildModel(training_config_001)
        model_base = "/path/to/model_1/nrt_bo_001/model_folder_1"
        out_base = "/path/to/build_1/nrt_bo_001/build_folder_1"

        for tgt in TARGETS:
            assert (
                str(ds.model_file_names[tgt])
                == f"{model_base}/model_{tgt}.joblib"
            )
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

    def test_training_sets(self, training_config_001, training_input_001):
        """training_sets and test_sets are loaded as DataFrames with expected shapes."""
        ds = BuildModel(
            training_config_001,
            training_sets=training_input_001.training_sets,
            test_sets=training_input_001.test_sets,
        )

        expected_train = {
            "temp": (116, 57),  # TODO: update to actual value after data reduction
            "psal": (126, 57),  # TODO: update to actual value after data reduction
            "pres": (110, 57),  # TODO: update to actual value after data reduction
        }
        expected_test = {
            "temp": (12, 56),  # TODO: update to actual value after data reduction
            "psal": (14, 56),  # TODO: update to actual value after data reduction
            "pres": (12, 56),  # TODO: update to actual value after data reduction
        }
        for tgt in TARGETS:
            assert isinstance(ds.training_sets[tgt], pl.DataFrame)
            assert ds.training_sets[tgt].shape == expected_train[tgt]
            assert isinstance(ds.test_sets[tgt], pl.DataFrame)
            assert ds.test_sets[tgt].shape == expected_test[tgt]

    def test_shap_flag(self, training_config_001):
        """``calculate_shap`` in config is forwarded to the base model.

        Unlike KFoldValidation (step2) which never enables SHAP, BuildModel
        honours the flag because SHAP is computed at the testing stage. This
        test defends that distinction: True flag -> enable_shap True.
        """
        ds = BuildModel(training_config_001)
        assert ds.base_model.enable_shap is False  # unset == False

        training_config_001.data["step_param_set"]["steps"]["model"]["calculate_shap"] = True
        ds = BuildModel(training_config_001)
        assert ds.base_model.enable_shap is True

        training_config_001.data["step_param_set"]["steps"]["model"]["calculate_shap"] = False
        ds = BuildModel(training_config_001)
        assert ds.base_model.enable_shap is False

    # ----- Default-config XGBoost path -----

    def test_train_with_xgboost(self, training_config_001, training_input_001):
        """Default config produces XGBoost instances in the .models dict."""
        ds = BuildModel(
            training_config_001,
            training_sets=training_input_001.training_sets,
            test_sets=training_input_001.test_sets,
        )
        ds.build_targets()

        for tgt in TARGETS:
            assert isinstance(ds.models[tgt], XGBoost)

    def test_train_final_model_with_xgboost(self, training_config_001, training_input_001):
        """Default config produces XGBoost instances in the .final_models dict."""
        ds = BuildModel(
            training_config_001,
            training_sets=training_input_001.training_sets,
            test_sets=training_input_001.test_sets,
        )
        ds.build_final_model_targets()

        for tgt in TARGETS:
            assert isinstance(ds.final_models[tgt], XGBoost)

    def test_read_models(self, training_config_001, training_input_001, training_dir):
        """Existing .joblib model fixtures can be loaded and used for testing."""
        ds = BuildModel(
            training_config_001,
            training_sets=None,
            test_sets=training_input_001.test_sets,
        )
        for tgt in TARGETS:
            ds.model_file_names[tgt] = str(training_dir / f"model_{tgt}.joblib")

        ds.read_models()

        for tgt in TARGETS:
            assert isinstance(ds.models[tgt], XGBoost)

        ds.test_targets()

        # After test_targets, two of the three test sets should be populated
        # (the original asserted only temp and psal; preserving that).
        expected_shapes = {
            "temp": (12, 56),  # TODO: update to actual value after data reduction
            "psal": (14, 56),  # TODO: update to actual value after data reduction
        }
        for tgt, expected in expected_shapes.items():
            assert isinstance(ds.test_sets[tgt], pl.DataFrame)
            assert ds.test_sets[tgt].shape == expected

    def test_read_models_no_file(self, training_config_001, training_input_001, training_dir):
        """Missing model files raise FileNotFoundError."""
        ds = BuildModel(
            training_config_001,
            training_sets=None,
            test_sets=training_input_001.test_sets,
        )
        for tgt in TARGETS:
            ds.model_file_names[tgt] = str(training_dir / "non_existent_model.joblib")

        with pytest.raises(FileNotFoundError):
            ds.read_models()

    # ----- Distinct-instance invariants -----

    def test_model_objects(self, training_config_001, training_input_001):
        """Each target's final_model is a distinct object (not a shared ref)."""
        ds = BuildModel(
            training_config_001,
            training_sets=training_input_001.training_sets,
            test_sets=training_input_001.test_sets,
        )
        ds.build_final_model_targets()

        assert ds.final_models["temp"] is not ds.final_models["psal"]
        assert ds.final_models["temp"] is not ds.final_models["pres"]
        assert ds.final_models["psal"] is not ds.final_models["pres"]

        # The assertNotEqual checks in the original depend on the wrapper's
        # __eq__ implementation, but assertIsNot is the stronger check we
        # actually care about. Preserved for behavioural parity.
        assert ds.final_models["temp"] != ds.final_models["psal"]
        assert ds.final_models["temp"] != ds.final_models["pres"]
        assert ds.final_models["psal"] != ds.final_models["pres"]

    def test_test_model_objects(self, training_config_001, training_input_001):
        """Each target's intermediate model is also a distinct object."""
        ds = BuildModel(
            training_config_001,
            training_sets=training_input_001.training_sets,
            test_sets=training_input_001.test_sets,
        )
        ds.build_targets()

        assert ds.models["temp"] is not ds.models["psal"]
        assert ds.models["temp"] is not ds.models["pres"]
        assert ds.models["psal"] is not ds.models["pres"]

        assert ds.models["temp"] != ds.models["psal"]
        assert ds.models["temp"] != ds.models["pres"]
        assert ds.models["psal"] != ds.models["pres"]

    # ----- Empty-state / error cases -----

    def test_build_without_training_sets(self, training_config_001):
        """build_targets with training_sets=None raises ValueError."""
        ds = BuildModel(training_config_001, training_sets=None, test_sets=None)
        with pytest.raises(ValueError):
            ds.build_targets()

    def test_build_final_model_without_test_sets(self, training_config_001, training_input_001):
        """build_final_model_targets with test_sets=None raises ValueError."""
        ds = BuildModel(
            training_config_001,
            training_sets=training_input_001.training_sets,
            test_sets=None,
        )
        with pytest.raises(ValueError):
            ds.build_final_model_targets()

    def test_build_final_model_without_training_sets(self, training_config_001):
        """build_final_model_targets with no inputs raises ValueError."""
        ds = BuildModel(training_config_001, training_sets=None, test_sets=None)
        with pytest.raises(ValueError):
            ds.build_final_model_targets()

    def test_test_without_model(self, training_config_001, training_input_001):
        """test_targets before build_targets raises ValueError."""
        ds = BuildModel(
            training_config_001,
            training_sets=training_input_001.training_sets,
            test_sets=training_input_001.test_sets,
        )
        with pytest.raises(ValueError):
            ds.test_targets()

    def test_write_no_results(self, training_config_001, training_input_001):
        """write_reports before test_targets raises ValueError."""
        ds = BuildModel(
            training_config_001,
            training_sets=training_input_001.training_sets,
            test_sets=training_input_001.test_sets,
        )
        with pytest.raises(ValueError):
            ds.write_reports()

    def test_write_empty_contingency_tables(self, training_config_001, training_input_001):
        """write_contingency_tables before test_targets raises ValueError."""
        ds = BuildModel(
            training_config_001,
            training_sets=training_input_001.training_sets,
            test_sets=training_input_001.test_sets,
        )
        with pytest.raises(ValueError):
            ds.write_contingency_tables()

    def test_create_empty_metric_plots(self, training_config_001, training_input_001):
        """create_metric_plots before test_targets raises ValueError."""
        ds = BuildModel(
            training_config_001,
            training_sets=training_input_001.training_sets,
            test_sets=training_input_001.test_sets,
        )
        with pytest.raises(ValueError):
            ds.create_metric_plots()

    def test_write_no_models(self, training_config_001, training_input_001):
        """write_models before build_final_model_targets raises ValueError."""
        ds = BuildModel(
            training_config_001,
            training_sets=training_input_001.training_sets,
            test_sets=training_input_001.test_sets,
        )
        with pytest.raises(ValueError):
            ds.write_models()

    def test_write_empty_predictions(self, training_config_001, training_input_001):
        """write_predictions before test_targets raises ValueError."""
        ds = BuildModel(
            training_config_001,
            training_sets=training_input_001.training_sets,
            test_sets=training_input_001.test_sets,
        )
        with pytest.raises(ValueError):
            ds.write_predictions()

    # ----- File output (write, assert exists, manually remove) -----

    def test_write_reports(self, training_config_001, training_input_001, test_output_dir):
        """write_reports produces a TSV per target at the configured path."""
        ds = BuildModel(
            training_config_001,
            training_sets=training_input_001.training_sets,
            test_sets=training_input_001.test_sets,
        )
        output_paths = {
            tgt: str(test_output_dir / f"test_test_report_{tgt}.tsv") for tgt in TARGETS
        }
        ds.output_file_names["report"] = output_paths

        ds.build_targets()
        ds.test_targets()
        ds.write_reports()

        for tgt in TARGETS:
            assert os.path.exists(output_paths[tgt])
            os.remove(output_paths[tgt])  # comment out to debug

    def test_write_contingency_tables(self, training_config_001, training_input_001, test_output_dir):
        """write_contingency_tables produces a parquet per target."""
        ds = BuildModel(
            training_config_001,
            training_sets=training_input_001.training_sets,
            test_sets=training_input_001.test_sets,
        )
        output_paths = {
            tgt: str(test_output_dir / f"test_test_contingency_{tgt}.parquet")
            for tgt in TARGETS
        }
        ds.output_file_names["contingency_table"] = output_paths

        ds.build_targets()
        ds.test_targets()
        ds.write_contingency_tables()

        for tgt in TARGETS:
            assert os.path.exists(output_paths[tgt])
            os.remove(output_paths[tgt])  # comment out to debug

    def test_write_shap_values(self, training_config_001, training_input_001, test_output_dir):
        """write_shap_values produces a parquet per target when calculate_shap=True."""
        training_config_001.data["step_param_set"]["steps"]["model"]["calculate_shap"] = True
        ds = BuildModel(
            training_config_001,
            training_sets=training_input_001.training_sets,
            test_sets=training_input_001.test_sets,
        )
        output_paths = {
            tgt: str(test_output_dir / f"test_test_shap_{tgt}.parquet") for tgt in TARGETS
        }
        ds.output_file_names["shap_value"] = output_paths

        ds.build_targets()
        ds.test_targets()
        ds.write_shap_values()

        for tgt in TARGETS:
            assert os.path.exists(output_paths[tgt])
            os.remove(output_paths[tgt])  # comment out to debug

    def test_create_metric_plots(self, training_config_001, training_input_001, test_output_dir):
        """create_metric_plots produces an SVG per target."""
        ds = BuildModel(
            training_config_001,
            training_sets=training_input_001.training_sets,
            test_sets=training_input_001.test_sets,
        )
        output_paths = {
            tgt: str(test_output_dir / f"test_test_metric_plots_{tgt}.svg") for tgt in TARGETS
        }
        ds.output_file_names["metric_plot"] = output_paths

        ds.build_targets()
        ds.test_targets()
        ds.create_metric_plots()

        for tgt in TARGETS:
            assert os.path.exists(output_paths[tgt])
            os.remove(output_paths[tgt])  # comment out to debug

    def test_write_models(self, training_config_001, training_input_001, test_output_dir):
        """write_models serialises a joblib per target."""
        ds = BuildModel(
            training_config_001,
            training_sets=training_input_001.training_sets,
            test_sets=training_input_001.test_sets,
        )
        output_paths = {
            tgt: str(test_output_dir / f"test_model_{tgt}.joblib") for tgt in TARGETS
        }
        ds.model_file_names = output_paths

        ds.build_final_model_targets()
        ds.write_models()

        for tgt in TARGETS:
            assert os.path.exists(output_paths[tgt])
            os.remove(output_paths[tgt])  # comment out to debug

    def test_write_predictions(self, training_config_001, training_input_001, test_output_dir):
        """write_predictions produces a parquet per target."""
        ds = BuildModel(
            training_config_001,
            training_sets=training_input_001.training_sets,
            test_sets=training_input_001.test_sets,
        )
        output_paths = {
            tgt: str(test_output_dir / f"test_test_prediction_{tgt}.parquet")
            for tgt in TARGETS
        }
        ds.output_file_names["prediction"] = output_paths

        ds.build_targets()
        ds.test_targets()
        ds.write_predictions()

        for tgt in TARGETS:
            assert os.path.exists(output_paths[tgt])
            os.remove(output_paths[tgt])  # comment out to debug


# ---------------------------------------------------------------------------
# Per-model fan-out
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", MODEL_CASES, ids=lambda c: c.config_name)
class TestModels:
    """Behavioural tests applied identically to every model wrapper.

    Replaces nine ``TestXGBoost``, ``TestLogisticRegression``, ..., classes,
    each of which had identical methods differing only in the model name.

    The parametrize id is the config_name, so ``pytest -k xgboost`` and
    similar still work.
    """

    def test_base_model(self, case, training_config_001):
        """BuildModel.base_model is an instance of the configured wrapper.

        NOTE: The original test in this file mistakenly constructed
        ``KFoldValidation`` here (a copy-paste from step2). Fixed to use
        ``BuildModel``, which is what the file under test actually exercises.
        """
        training_config_001.data["step_class_set"]["steps"]["model"] = case.config_name
        ds = BuildModel(training_config_001)
        assert isinstance(ds.base_model, case.wrapper_cls)

    def test_training(self, case, training_config_001, training_input_001):
        """build_targets populates .models with instances of the configured wrapper."""
        training_config_001.data["step_class_set"]["steps"]["model"] = case.config_name
        ds = BuildModel(
            training_config_001,
            training_sets=training_input_001.training_sets,
            test_sets=training_input_001.test_sets,
        )
        ds.build_targets()

        for tgt in TARGETS:
            assert isinstance(ds.models[tgt], case.wrapper_cls)

    def test_model_output(self, case, training_config_001, training_input_001):
        """End-to-end build + test with the configured model produces correct outputs."""
        training_config_001.data["step_class_set"]["steps"]["model"] = case.config_name
        ds = BuildModel(
            training_config_001,
            training_sets=training_input_001.training_sets,
            test_sets=training_input_001.test_sets,
        )
        _run_test_with_trained_model(ds)

    def test_write_model(self, case, training_config_001, training_input_001, test_output_dir):
        """write_models produces a joblib per target with a per-model filename."""
        training_config_001.data["step_class_set"]["steps"]["model"] = case.config_name
        ds = BuildModel(
            training_config_001,
            training_sets=training_input_001.training_sets,
            test_sets=training_input_001.test_sets,
        )
        output_paths = {
            tgt: str(test_output_dir / f"test_model_{tgt}_{case.joblib_suffix}.joblib")
            for tgt in TARGETS
        }
        ds.model_file_names = output_paths

        ds.build_final_model_targets()
        ds.write_models()

        for tgt in TARGETS:
            assert os.path.exists(output_paths[tgt])
            os.remove(output_paths[tgt])  # comment out to debug