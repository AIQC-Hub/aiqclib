"""Unit tests for the ``ClassifyAllSuite`` class.

ClassifyAllSuite is the multi-model variant of ClassifyAll: it loads several
trained models in parallel (here XGB and DT, for speed) and produces
aggregated outputs with a ``method`` column distinguishing per-model
predictions.

Refactored from the original which:
- Used two module-level helpers ``_setup_datasets(test_obj)`` and
  ``_setup_classify_all_suite(test_obj)`` to inject ModelSuite settings and
  then run the full prepare pipeline (replaced here by a ``mutate_config``
  callback passed to ``run_classify_prepare_pipeline`` in conftest)
- Used ``TEST_COUNT = 1`` but kept three configs available — preserved here
  (the ``classify_suite_pipeline_first`` fixture uses only config 001).
- Triplicated per-target paths/assertions in every test (replaced with
  per-TARGETS loops)

No 9-model fan-out here: ClassifyAllSuite tests the *combined* behaviour of
multiple methods running together, not each model wrapper individually.
"""

import os
from types import SimpleNamespace

import matplotlib
import polars as pl
import pytest

# Non-interactive backend so plot tests don't try to open windows.
matplotlib.use("Agg")

from aiqclib.classify.step6_classify_dataset.dataset_all_suite import ClassifyAllSuite
from aiqclib.common.config.classify_config import ClassificationConfig
from aiqclib.train.models.model_suite import ModelSuite

from tests.conftest import TARGETS_NONEMPTY, run_classify_prepare_pipeline


# Composite keys produced by ModelSuite when methods=["XGB", "DT"]:
# 2 methods × 2 targets = 4 keys.
SUITE_METHODS = ("xgb", "dt")
SUITE_KEYS = tuple(f"{method}_{tgt}" for method in SUITE_METHODS for tgt in TARGETS_NONEMPTY)


# ---------------------------------------------------------------------------
# Suite-mutation helper (passed to run_classify_prepare_pipeline)
# ---------------------------------------------------------------------------

def _inject_suite_settings(config: ClassificationConfig) -> None:
    """Mutate a ClassificationConfig to use ClassifyAllSuite + ModelSuite + 2 methods.

    Applied via the ``mutate_config`` callback of
    ``run_classify_prepare_pipeline`` so the prepare steps see the
    suite-configured state. Three keys touched:
    - step_class_set.steps.classify = ClassifyAllSuite
    - step_class_set.steps.model = ModelSuite
    - step_param_set.steps.model.methods = [XGB, DT]
    """
    config.data["step_class_set"]["steps"]["classify"] = "ClassifyAllSuite"
    config.data["step_class_set"]["steps"]["model"] = "ModelSuite"
    config.data["step_param_set"]["steps"]["model"] = {"methods": ["XGB", "DT"]}


# ---------------------------------------------------------------------------
# Suite-config fixture (static tests; no pipeline needed)
# ---------------------------------------------------------------------------

@pytest.fixture
def classify_config_001_suite(classify_config_001):
    """classify_config_001 with suite settings injected."""
    _inject_suite_settings(classify_config_001)
    return classify_config_001


# ---------------------------------------------------------------------------
# Pipeline-driven fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def classify_suite_pipeline_first(test_data_file, classify_yaml_001):
    """Run the prepare pipeline (with suite mutations) against config 001 only.

    The original used three config paths but TEST_COUNT=1, so only config 001
    was ever exercised. Preserved here.
    """
    configs, extracts = run_classify_prepare_pipeline(
        [classify_yaml_001],
        test_data_file,
        mutate_config=_inject_suite_settings,
    )
    return SimpleNamespace(configs=configs, extracts=extracts)


# ---------------------------------------------------------------------------
# Model file paths (composite keys, shared by all pipeline tests)
# ---------------------------------------------------------------------------

@pytest.fixture
def suite_model_files(training_dir):
    """Composite-key model file paths used by ClassifyAllSuite tests.

    The original's comment noted: each composite key maps to the existing
    single-model fixture for that algorithm. ``xgb_*`` keys point at
    ``model_*_xgb.joblib``, ``dt_*`` at ``model_*_dt.joblib``. This works
    because read_models loads by composite-key path independently — there's
    no requirement that all six files contain coherent multi-method data.
    """
    return {
        f"{method}_{tgt}": str(training_dir / f"model_{tgt}_{method}.joblib")
        for method in SUITE_METHODS
        for tgt in TARGETS_NONEMPTY
    }


# ---------------------------------------------------------------------------
# Static tests (no pipeline output required)
# ---------------------------------------------------------------------------

class TestClassifyAllSuiteClass:
    """Tests against ClassifyAllSuite that don't need the prepare pipeline.

    These verify identity, default output paths, base_model wiring, and the
    multi-flag check. None of these touch actual data, so the pipeline isn't
    run for them.
    """

    def test_step_name(self, classify_config_001_suite):
        """step_name == 'classify'."""
        ds = ClassifyAllSuite(classify_config_001_suite)
        assert ds.step_name == "classify"

    def test_multi_flag_check(self, classify_config_001_suite):
        """Constructing ClassifyAllSuite with a single-model wrapper raises ValueError.

        Same invariant as BuildModelSuite: the error message must mention
        ``multi=True`` so the user knows what to fix.
        """
        classify_config_001_suite.data["step_class_set"]["steps"]["model"] = "XGBoost"
        with pytest.raises(ValueError, match="multi=True"):
            _ = ClassifyAllSuite(classify_config_001_suite)

    def test_output_file_names(self, classify_config_001_suite):
        """Model files use composite ``{method}_{tgt}`` keys; outputs use ``{tgt}``."""
        ds = ClassifyAllSuite(classify_config_001_suite)
        model_base = "/path/to/model_1/model_folder_1"
        out_base = "/path/to/classify_1/nrt_bo_001/classify_folder_1"

        # Model files: composite keys
        for method in SUITE_METHODS:
            for tgt in TARGETS_NONEMPTY:
                assert (
                    str(ds.model_file_names[f"{method}_{tgt}"])
                    == f"{model_base}/model_{method}_{tgt}.joblib"
                )

        # Aggregated output files: target keys (no method in name)
        for tgt in TARGETS_NONEMPTY:
            assert (
                str(ds.output_file_names["report"][tgt])
                == f"{out_base}/classify_report_{tgt}.tsv"
            )
            assert (
                str(ds.output_file_names["contingency_table"][tgt])
                == f"{out_base}/classify_contingency_tables_{tgt}.parquet"
            )
            assert (
                str(ds.output_file_names["metric_plot"][tgt])
                == f"{out_base}/classify_metric_plots_{tgt}.svg"
            )

    def test_base_model(self, classify_config_001_suite):
        """The base_model is a ModelSuite instance."""
        ds = ClassifyAllSuite(classify_config_001_suite)
        assert isinstance(ds.base_model, ModelSuite)


# ---------------------------------------------------------------------------
# Pipeline-driven tests
# ---------------------------------------------------------------------------

class TestClassifyAllSuite:
    """Pipeline-driven tests against ClassifyAllSuite.

    The original parametrized these over ``idx ∈ range(TEST_COUNT)`` with
    TEST_COUNT=1 — effectively a single test per method. Preserved as
    single-config tests (no parametrize needed).
    """

    def test_test_sets(self, classify_suite_pipeline_first):
        """test_sets are loaded with the expected per-target shape."""
        ds = ClassifyAllSuite(
            classify_suite_pipeline_first.configs[0],
            test_sets=classify_suite_pipeline_first.extracts[0].target_features,
        )
        assert isinstance(ds.test_sets["temp"], pl.DataFrame)
        assert ds.test_sets["temp"].shape[0] == 2456
        assert ds.test_sets["temp"].shape[1] == 56

    def test_read_models(self, classify_suite_pipeline_first, suite_model_files):
        """read_models populates .models with all six composite keys."""
        ds = ClassifyAllSuite(
            classify_suite_pipeline_first.configs[0],
            test_sets=classify_suite_pipeline_first.extracts[0].target_features,
        )
        ds.model_file_names = suite_model_files
        ds.read_models()

        for key in SUITE_KEYS:
            assert key in ds.models

    def test_shap_flag(self, classify_suite_pipeline_first):
        """``calculate_shap`` propagates to both the suite and each child method.

        Same invariant as BuildModelSuite: the suite-level flag and each
        per-method flag move together.
        """
        config = classify_suite_pipeline_first.configs[0]

        ds = ClassifyAllSuite(config)
        assert ds.base_model.enable_shap is False
        for method_obj in ds.base_model.method_objs.values():
            assert method_obj.enable_shap is False

        config.data["step_param_set"]["steps"]["model"]["calculate_shap"] = True
        ds = ClassifyAllSuite(config)
        assert ds.base_model.enable_shap is True
        for method_obj in ds.base_model.method_objs.values():
            assert method_obj.enable_shap is True

        config.data["step_param_set"]["steps"]["model"]["calculate_shap"] = False
        ds = ClassifyAllSuite(config)
        assert ds.base_model.enable_shap is False
        for method_obj in ds.base_model.method_objs.values():
            assert method_obj.enable_shap is False

    def test_with_models(self, classify_suite_pipeline_first, suite_model_files):
        """End-to-end: read_models + test_targets aggregates per-method predictions.

        Predictions and contingency tables include a 'method' column, and
        their row count is (test_set rows × number of methods).
        """
        ds = ClassifyAllSuite(
            classify_suite_pipeline_first.configs[0],
            test_sets=classify_suite_pipeline_first.extracts[0].target_features,
        )
        ds.model_file_names = suite_model_files
        ds.read_models()
        ds.test_targets()

        # 812 test rows × 2 methods = 4912. After data reduction this
        # becomes (new_test_size × 2).
        expected_aggregated_height = 4912

        # predictions
        assert isinstance(ds.predictions["temp"], pl.DataFrame)
        assert ds.predictions["temp"].shape[0] == expected_aggregated_height
        assert "method" in ds.predictions["temp"].columns

        # contingency_tables
        assert isinstance(ds.contingency_tables["psal"], pl.DataFrame)
        assert ds.contingency_tables["psal"].height == expected_aggregated_height
        assert "method" in ds.contingency_tables["psal"].columns

    # ----- Error cases -----

    def test_without_model(self, classify_suite_pipeline_first):
        """test_targets without loaded models raises ValueError."""
        ds = ClassifyAllSuite(
            classify_suite_pipeline_first.configs[0],
            test_sets=classify_suite_pipeline_first.extracts[0].target_features,
        )
        with pytest.raises(ValueError):
            ds.test_targets()

    def test_write_no_results(self, classify_suite_pipeline_first):
        """write_reports before test_targets raises ValueError."""
        ds = ClassifyAllSuite(
            classify_suite_pipeline_first.configs[0],
            test_sets=classify_suite_pipeline_first.extracts[0].target_features,
        )
        with pytest.raises(ValueError):
            ds.write_reports()

    def test_read_models_no_file(self, classify_suite_pipeline_first, training_dir):
        """Missing model files raise FileNotFoundError."""
        ds = ClassifyAllSuite(
            classify_suite_pipeline_first.configs[0],
            test_sets=classify_suite_pipeline_first.extracts[0].target_features,
        )
        ds.model_file_names["xgb_temp"] = str(training_dir / "model_does_not_exist.joblib")

        with pytest.raises(FileNotFoundError):
            ds.read_models()

    # ----- File output (write, assert per-target exists, manually remove) -----

    def test_write_reports(
        self, classify_suite_pipeline_first, suite_model_files, test_output_dir
    ):
        """write_reports produces an aggregated TSV per target."""
        ds = ClassifyAllSuite(
            classify_suite_pipeline_first.configs[0],
            test_sets=classify_suite_pipeline_first.extracts[0].target_features,
        )
        ds.model_file_names = suite_model_files
        output_paths = {
            tgt: str(test_output_dir / f"test_classify_suite_report_{tgt}.tsv")
            for tgt in TARGETS_NONEMPTY
        }
        ds.output_file_names["report"] = output_paths

        ds.read_models()
        ds.test_targets()
        ds.write_reports()

        for tgt in TARGETS_NONEMPTY:
            assert os.path.exists(output_paths[tgt])
            os.remove(output_paths[tgt])  # comment out to debug

    def test_write_contingency_tables(
        self, classify_suite_pipeline_first, suite_model_files, test_output_dir
    ):
        """write_contingency_tables produces an aggregated parquet per target."""
        ds = ClassifyAllSuite(
            classify_suite_pipeline_first.configs[0],
            test_sets=classify_suite_pipeline_first.extracts[0].target_features,
        )
        ds.model_file_names = suite_model_files
        output_paths = {
            tgt: str(test_output_dir / f"test_classify_suite_contingency_{tgt}.parquet")
            for tgt in TARGETS_NONEMPTY
        }
        ds.output_file_names["contingency_table"] = output_paths

        ds.read_models()
        ds.test_targets()
        ds.write_contingency_tables()

        for tgt in TARGETS_NONEMPTY:
            assert os.path.exists(output_paths[tgt])
            os.remove(output_paths[tgt])  # comment out to debug

    def test_write_shap_values(
        self, classify_suite_pipeline_first, suite_model_files, test_output_dir
    ):
        """write_shap_values produces an aggregated parquet per target."""
        classify_suite_pipeline_first.configs[0].data["step_param_set"]["steps"]["model"][
            "calculate_shap"
        ] = True
        ds = ClassifyAllSuite(
            classify_suite_pipeline_first.configs[0],
            test_sets=classify_suite_pipeline_first.extracts[0].target_features,
        )
        ds.model_file_names = suite_model_files
        output_paths = {
            tgt: str(test_output_dir / f"test_classify_suite_shap_{tgt}.parquet")
            for tgt in TARGETS_NONEMPTY
        }
        ds.output_file_names["shap_value"] = output_paths

        ds.read_models()
        ds.test_targets()
        ds.write_shap_values()

        for tgt in TARGETS_NONEMPTY:
            assert os.path.exists(output_paths[tgt])
            os.remove(output_paths[tgt])  # comment out to debug

    def test_create_metric_plot(
        self, classify_suite_pipeline_first, suite_model_files, test_output_dir
    ):
        """create_metric_plots produces a multi-method SVG per target."""
        ds = ClassifyAllSuite(
            classify_suite_pipeline_first.configs[0],
            test_sets=classify_suite_pipeline_first.extracts[0].target_features,
        )
        ds.model_file_names = suite_model_files
        output_paths = {
            tgt: str(test_output_dir / f"test_classify_suite_metric_plots_{tgt}.svg")
            for tgt in TARGETS_NONEMPTY
        }
        ds.output_file_names["metric_plot"] = output_paths

        ds.read_models()
        ds.test_targets()
        ds.create_metric_plots()

        for tgt in TARGETS_NONEMPTY:
            assert os.path.exists(output_paths[tgt])
            os.remove(output_paths[tgt])  # comment out to debug

    def test_write_predictions(
        self, classify_suite_pipeline_first, suite_model_files, test_output_dir
    ):
        """write_predictions produces an aggregated parquet per target."""
        ds = ClassifyAllSuite(
            classify_suite_pipeline_first.configs[0],
            test_sets=classify_suite_pipeline_first.extracts[0].target_features,
        )
        ds.model_file_names = suite_model_files
        output_paths = {
            tgt: str(test_output_dir / f"test_classify_suite_prediction_{tgt}.parquet")
            for tgt in TARGETS_NONEMPTY
        }
        ds.output_file_names["prediction"] = output_paths

        ds.read_models()
        ds.test_targets()
        ds.write_predictions()

        for tgt in TARGETS_NONEMPTY:
            assert os.path.exists(output_paths[tgt])
            os.remove(output_paths[tgt])  # comment out to debug