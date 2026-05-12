"""Unit tests for the ``ClassifyAll`` class.

Exercises:
- Identity, config wiring (step_name, output_file_names), base_model wiring
- shap_flag forwarding (ClassifyAll honours the flag, like BuildModel does)
- Default-XGBoost classification path: read default models, test_targets,
  write reports/contingency tables/SHAP values/predictions, create metric
  plots — plus all their empty-state ValueError counterparts.
- Per-config behaviour: the three test_classify_*.yaml configs are exercised
  in parallel via the ``idx`` parametrize; ``test_n_jobs`` verifies each
  config's configured thread count is honoured.
- The 9-model fan-out, previously nine sibling classes, now a single
  parametrized ``TestModels`` class importing MODEL_CASES.

Refactored from the original which had:
- Two module-level helpers ``_setup_datasets(test_obj)`` and
  ``_setup_classify_all(test_obj)`` (replaced by ``classify_pipeline_all``
  and ``classify_pipeline_first`` fixtures in this file — kept local since
  they're specific to classify-stage tests)
- ~30 lines of per-target triplication per test (now loops over ``TARGETS``)
- Nine ``Test{Model}`` classes (collapsed into ``TestModels`` parametrized
  by MODEL_CASES)
"""

import os
from types import SimpleNamespace

import polars as pl
import pytest

from aiqclib.classify.step6_classify_dataset.dataset_all import ClassifyAll
from aiqclib.common.config.classify_config import ClassificationConfig
from aiqclib.common.loader.classify_loader import (
    load_classify_step1_input_dataset,
    load_classify_step2_summary_dataset,
    load_classify_step3_select_dataset,
    load_classify_step4_locate_dataset,
    load_classify_step5_extract_dataset,
)
from aiqclib.train.models.logistic_regression import LogisticRegression
from aiqclib.train.models.xgboost import XGBoost

from tests._model_cases import MODEL_CASES
from tests.conftest import TARGETS


# ---------------------------------------------------------------------------
# Test-file-specific constants
# ---------------------------------------------------------------------------

# n_jobs values that each test_classify_*.yaml configures. Used by
# test_n_jobs to verify the YAML is honoured.
N_JOBS_PER_CONFIG = [-1, -1, 2]


# ---------------------------------------------------------------------------
# Prepare-pipeline helper (specific to classify tests; kept here rather than
# conftest because no other test stage uses it)
# ---------------------------------------------------------------------------

def _run_prepare_pipeline(config_files, test_data_file):
    """Run prepare steps 1-5 for each config and return paired (configs, extracts).

    Each ClassifyAll test needs a fully-prepared pipeline output (the
    ``extracts[idx].target_features`` is the test set ClassifyAll classifies).
    The original file had this as two module-level helpers; consolidated
    here into one function that returns a (configs, extracts) pair, wrapped
    in a SimpleNamespace by the calling fixture.
    """
    configs = []
    extracts = []
    for path in config_files:
        config = ClassificationConfig(str(path))
        config.select("NRT_BO_001")

        ds_input = load_classify_step1_input_dataset(config)
        ds_input.input_file_name = str(test_data_file)
        ds_input.read_input_data()

        ds_summary = load_classify_step2_summary_dataset(
            config, input_data=ds_input.input_data
        )
        ds_summary.calculate_stats()

        ds_select = load_classify_step3_select_dataset(
            config, input_data=ds_input.input_data
        )
        ds_select.label_profiles()

        ds_locate = load_classify_step4_locate_dataset(
            config,
            input_data=ds_input.input_data,
            selected_profiles=ds_select.selected_profiles,
        )
        ds_locate.process_targets()

        ds_extract = load_classify_step5_extract_dataset(
            config,
            input_data=ds_input.input_data,
            selected_profiles=ds_select.selected_profiles,
            selected_rows=ds_locate.selected_rows,
            summary_stats=ds_summary.summary_stats,
        )
        ds_extract.process_targets()

        configs.append(config)
        extracts.append(ds_extract)

    return configs, extracts


# ---------------------------------------------------------------------------
# Fixtures specific to this test file
# ---------------------------------------------------------------------------

@pytest.fixture
def classify_pipeline_all(
    test_data_file, classify_yaml_001, classify_yaml_002, classify_yaml_003
):
    """Run the prepare pipeline against all three classify configs.

    Used by TestClassifyAll, whose tests parametrize over idx ∈ {0, 1, 2}.
    Returns a SimpleNamespace with ``configs`` and ``extracts`` lists, each
    of length 3 in the same order.
    """
    configs, extracts = _run_prepare_pipeline(
        [classify_yaml_001, classify_yaml_002, classify_yaml_003],
        test_data_file,
    )
    return SimpleNamespace(configs=configs, extracts=extracts)


@pytest.fixture
def classify_pipeline_first(test_data_file, classify_yaml_001):
    """Run the prepare pipeline against only test_classify_001.yaml.

    Used by TestModels, whose tests don't parametrize over configs.
    Returns the same SimpleNamespace shape as classify_pipeline_all but
    with one-element lists.
    """
    configs, extracts = _run_prepare_pipeline([classify_yaml_001], test_data_file)
    return SimpleNamespace(configs=configs, extracts=extracts)


@pytest.fixture
def default_model_files(training_dir):
    """Default XGBoost model fixture paths used by TestClassifyAll tests.

    These are the unsuffixed ``model_{tgt}.joblib`` files — the default
    XGBoost variant. Per-model tests in TestModels use suffixed files
    constructed from ``case.joblib_suffix``.
    """
    return {tgt: str(training_dir / f"model_{tgt}.joblib") for tgt in TARGETS}


# ---------------------------------------------------------------------------
# Static tests (no pipeline setup required)
# ---------------------------------------------------------------------------

class TestClassifyAllClass:
    """Tests against ClassifyAll that don't need the prepare pipeline output.

    These exercise the class's identity, default output paths, base_model
    wiring, and shap-flag/n_jobs config plumbing. Nothing here touches actual
    data, so we use the lightweight classify_config_001 / classify_config_003
    fixtures rather than running the full prepare pipeline.
    """

    def test_step_name(self, classify_config_001):
        """step_name == 'classify'."""
        ds = ClassifyAll(classify_config_001)
        assert ds.step_name == "classify"

    def test_output_file_names(self, classify_config_001):
        """Default output paths derive from config.path_info; no data-dependence."""
        ds = ClassifyAll(classify_config_001)
        model_base = "/path/to/model_1/model_folder_1"
        out_base = "/path/to/classify_1/nrt_bo_001/classify_folder_1"

        for tgt in TARGETS:
            assert str(ds.model_file_names[tgt]) == f"{model_base}/model_{tgt}.joblib"
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

    def test_base_model(self, classify_config_001):
        """Default config selects XGBoost as the base model."""
        ds = ClassifyAll(classify_config_001)
        assert isinstance(ds.base_model, XGBoost)

    def test_shap_flag(self, classify_config_001):
        """``calculate_shap`` in config is forwarded to the base model.

        Like BuildModel (step4), ClassifyAll honours the flag because SHAP
        is computed at the testing stage.
        """
        ds = ClassifyAll(classify_config_001)
        assert ds.base_model.enable_shap is False

        classify_config_001.data["step_param_set"]["steps"]["model"]["calculate_shap"] = True
        ds = ClassifyAll(classify_config_001)
        assert ds.base_model.enable_shap is True

        classify_config_001.data["step_param_set"]["steps"]["model"]["calculate_shap"] = False
        ds = ClassifyAll(classify_config_001)
        assert ds.base_model.enable_shap is False

    def test_logistic_regression_model(self, classify_config_001):
        """Setting model=LogisticRegression in config produces a LogisticRegression base."""
        classify_config_001.data["step_class_set"]["steps"]["model"] = "LogisticRegression"
        ds = ClassifyAll(classify_config_001)
        assert isinstance(ds.base_model, LogisticRegression)

    def test_nthreads(self, classify_config_001, classify_config_003):
        """``n_jobs`` value from YAML reaches the base model's model_params."""
        ds_1 = ClassifyAll(classify_config_001)
        assert ds_1.base_model.model_params["n_jobs"] == -1

        ds_3 = ClassifyAll(classify_config_003)
        assert ds_3.base_model.model_params["n_jobs"] == 2


# ---------------------------------------------------------------------------
# Pipeline-driven tests (default XGBoost path, parametrized over 3 configs)
# ---------------------------------------------------------------------------

class TestClassifyAll:
    """Pipeline-driven tests against all three classify configs.

    Each test method is parametrized over ``idx ∈ {0, 1, 2}``, corresponding
    to test_classify_001.yaml, test_classify_002.yaml, and
    test_classify_003.yaml. The fixture ``classify_pipeline_all`` provides
    the configs and extracts; each method picks the idx'th element.
    """

    @pytest.mark.parametrize("idx", range(3))
    def test_test_sets(self, idx, classify_pipeline_all):
        """test_sets are loaded into ClassifyAll with expected per-target shapes."""
        ds = ClassifyAll(
            classify_pipeline_all.configs[idx],
            test_sets=classify_pipeline_all.extracts[idx].target_features,
        )

        for tgt in TARGETS:
            assert isinstance(ds.test_sets[tgt], pl.DataFrame)
            # TODO: update to actual value after data reduction (all three
            # targets had shape (812, 56) on the original dataset).
            assert ds.test_sets[tgt].shape[0] == 812
            assert ds.test_sets[tgt].shape[1] == 56

    @pytest.mark.parametrize("idx", range(3))
    def test_read_models(self, idx, classify_pipeline_all, default_model_files):
        """read_models populates .models with XGBoost instances from default fixtures."""
        ds = ClassifyAll(
            classify_pipeline_all.configs[idx],
            test_sets=classify_pipeline_all.extracts[idx].target_features,
        )
        ds.model_file_names = default_model_files
        ds.read_models()

        for tgt in TARGETS:
            assert isinstance(ds.models[tgt], XGBoost)

    @pytest.mark.parametrize("idx", range(3))
    def test_with_xgboost(self, idx, classify_pipeline_all, default_model_files):
        """End-to-end: read_models + test_targets populates test_sets + contingency_tables."""
        ds = ClassifyAll(
            classify_pipeline_all.configs[idx],
            test_sets=classify_pipeline_all.extracts[idx].target_features,
        )
        ds.model_file_names = default_model_files
        ds.read_models()
        ds.test_targets()

        for tgt in TARGETS:
            assert isinstance(ds.test_sets[tgt], pl.DataFrame)
            # TODO: update to actual value after data reduction
            assert ds.test_sets[tgt].shape[0] == 812
            assert ds.test_sets[tgt].shape[1] == 56

            assert isinstance(ds.contingency_tables[tgt], pl.DataFrame)
            # TODO: update to actual value after data reduction
            assert ds.contingency_tables[tgt].height == 812

        assert ds.contingency_tables["temp"].columns == [
            "k", "label", "predicted_label", "score",
        ]

    @pytest.mark.parametrize("idx", range(3))
    def test_without_model(self, idx, classify_pipeline_all):
        """test_targets without loaded models raises ValueError."""
        ds = ClassifyAll(
            classify_pipeline_all.configs[idx],
            test_sets=classify_pipeline_all.extracts[idx].target_features,
        )
        with pytest.raises(ValueError):
            ds.test_targets()

    @pytest.mark.parametrize("idx", range(3))
    def test_write_reports(
        self, idx, classify_pipeline_all, default_model_files, test_output_dir
    ):
        """write_reports produces a TSV per target."""
        ds = ClassifyAll(
            classify_pipeline_all.configs[idx],
            test_sets=classify_pipeline_all.extracts[idx].target_features,
        )
        ds.model_file_names = default_model_files
        output_paths = {
            tgt: str(test_output_dir / f"test_classify_report_{tgt}.tsv")
            for tgt in TARGETS
        }
        ds.output_file_names["report"] = output_paths

        ds.read_models()
        ds.test_targets()
        ds.write_reports()

        for tgt in TARGETS:
            assert os.path.exists(output_paths[tgt])
            os.remove(output_paths[tgt])  # comment out to debug

    @pytest.mark.parametrize("idx", range(3))
    def test_write_contingency_tables(
        self, idx, classify_pipeline_all, default_model_files, test_output_dir
    ):
        """write_contingency_tables produces a parquet per target."""
        ds = ClassifyAll(
            classify_pipeline_all.configs[idx],
            test_sets=classify_pipeline_all.extracts[idx].target_features,
        )
        ds.model_file_names = default_model_files
        output_paths = {
            tgt: str(test_output_dir / f"test_classify_contingency_{tgt}.parquet")
            for tgt in TARGETS
        }
        ds.output_file_names["contingency_table"] = output_paths

        ds.read_models()
        ds.test_targets()
        ds.write_contingency_tables()

        for tgt in TARGETS:
            assert os.path.exists(output_paths[tgt])
            os.remove(output_paths[tgt])  # comment out to debug

    @pytest.mark.parametrize("idx", range(3))
    def test_write_shap_values(
        self, idx, classify_pipeline_all, default_model_files, test_output_dir
    ):
        """write_shap_values produces a parquet per target when calculate_shap=True."""
        classify_pipeline_all.configs[idx].data["step_param_set"]["steps"]["model"][
            "calculate_shap"
        ] = True
        ds = ClassifyAll(
            classify_pipeline_all.configs[idx],
            test_sets=classify_pipeline_all.extracts[idx].target_features,
        )
        ds.model_file_names = default_model_files
        output_paths = {
            tgt: str(test_output_dir / f"test_classify_shap_{tgt}.parquet")
            for tgt in TARGETS
        }
        ds.output_file_names["shap_value"] = output_paths

        ds.read_models()
        ds.test_targets()
        ds.write_shap_values()

        for tgt in TARGETS:
            assert os.path.exists(output_paths[tgt])
            os.remove(output_paths[tgt])  # comment out to debug

    @pytest.mark.parametrize("idx", range(3))
    def test_create_metric_plot(
        self, idx, classify_pipeline_all, default_model_files, test_output_dir
    ):
        """create_metric_plots produces an SVG per target."""
        ds = ClassifyAll(
            classify_pipeline_all.configs[idx],
            test_sets=classify_pipeline_all.extracts[idx].target_features,
        )
        ds.model_file_names = default_model_files
        output_paths = {
            tgt: str(test_output_dir / f"test_classify_metric_plots_{tgt}.svg")
            for tgt in TARGETS
        }
        ds.output_file_names["metric_plot"] = output_paths

        ds.read_models()
        ds.test_targets()
        ds.create_metric_plots()

        for tgt in TARGETS:
            assert os.path.exists(output_paths[tgt])
            os.remove(output_paths[tgt])  # comment out to debug

    @pytest.mark.parametrize("idx", range(3))
    def test_write_no_results(self, idx, classify_pipeline_all):
        """write_reports before test_targets raises ValueError."""
        ds = ClassifyAll(
            classify_pipeline_all.configs[idx],
            test_sets=classify_pipeline_all.extracts[idx].target_features,
        )
        with pytest.raises(ValueError):
            ds.write_reports()

    @pytest.mark.parametrize("idx", range(3))
    def test_write_no_contingency_tables(self, idx, classify_pipeline_all):
        """write_contingency_tables before test_targets raises ValueError."""
        ds = ClassifyAll(
            classify_pipeline_all.configs[idx],
            test_sets=classify_pipeline_all.extracts[idx].target_features,
        )
        with pytest.raises(ValueError):
            ds.write_contingency_tables()

    @pytest.mark.parametrize("idx", range(3))
    def test_create_no_metric_plots(self, idx, classify_pipeline_all):
        """create_metric_plots before test_targets raises ValueError."""
        ds = ClassifyAll(
            classify_pipeline_all.configs[idx],
            test_sets=classify_pipeline_all.extracts[idx].target_features,
        )
        with pytest.raises(ValueError):
            ds.create_metric_plots()

    @pytest.mark.parametrize("idx", range(3))
    def test_read_models_no_file(self, idx, classify_pipeline_all, training_dir):
        """Missing model files raise FileNotFoundError."""
        ds = ClassifyAll(
            classify_pipeline_all.configs[idx],
            test_sets=classify_pipeline_all.extracts[idx].target_features,
        )
        for tgt in TARGETS:
            ds.model_file_names[tgt] = str(training_dir / "model.joblib")

        with pytest.raises(FileNotFoundError):
            ds.read_models()

    @pytest.mark.parametrize("idx", range(3))
    def test_n_jobs(self, idx, classify_pipeline_all, default_model_files):
        """The configured n_jobs reaches both model_params and the underlying estimator."""
        ds = ClassifyAll(
            classify_pipeline_all.configs[idx],
            test_sets=classify_pipeline_all.extracts[idx].target_features,
        )
        ds.model_file_names = default_model_files
        ds.read_models()

        expected_n_jobs = N_JOBS_PER_CONFIG[idx]
        for tgt in TARGETS:
            assert ds.models[tgt].model_params["n_jobs"] == expected_n_jobs
            assert ds.models[tgt].model.n_jobs == expected_n_jobs

    @pytest.mark.parametrize("idx", range(3))
    def test_write_predictions(
        self, idx, classify_pipeline_all, default_model_files, test_output_dir
    ):
        """write_predictions produces a parquet per target."""
        ds = ClassifyAll(
            classify_pipeline_all.configs[idx],
            test_sets=classify_pipeline_all.extracts[idx].target_features,
        )
        ds.model_file_names = default_model_files
        output_paths = {
            tgt: str(test_output_dir / f"test_classify_prediction_{tgt}.parquet")
            for tgt in TARGETS
        }
        ds.output_file_names["prediction"] = output_paths

        ds.read_models()
        ds.test_targets()
        ds.write_predictions()

        for tgt in TARGETS:
            assert os.path.exists(output_paths[tgt])
            os.remove(output_paths[tgt])  # comment out to debug


# ---------------------------------------------------------------------------
# Per-model fan-out (parametrized over MODEL_CASES, single config)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", MODEL_CASES, ids=lambda c: c.config_name)
class TestModels:
    """Per-model classification behavioural tests.

    Replaces nine ``TestXGBoost``/``TestLogisticRegression``/... classes,
    each with two methods (``test_read_models`` and ``test_with_<algo>``).
    Each per-model class read its model from ``model_{tgt}_{suffix}.joblib``
    where the suffix matched the algorithm (e.g. ``_xgb``, ``_logit``,
    ``_lda``). The MODEL_CASES table now carries that suffix in
    ``case.joblib_suffix``.

    Only config 001 is tested (the original's METHOD_TEST_COUNT=1).
    """

    def test_read_models(self, case, classify_pipeline_first, training_dir):
        """read_models loads the per-model joblib and produces case.wrapper_cls instances."""
        config = classify_pipeline_first.configs[0]
        config.data["step_class_set"]["steps"]["model"] = case.config_name

        ds = ClassifyAll(
            config,
            test_sets=classify_pipeline_first.extracts[0].target_features,
        )
        ds.model_file_names = {
            tgt: str(training_dir / f"model_{tgt}_{case.joblib_suffix}.joblib")
            for tgt in TARGETS
        }
        ds.read_models()

        for tgt in TARGETS:
            assert isinstance(ds.models[tgt], case.wrapper_cls)

    def test_classify_with_model(self, case, classify_pipeline_first, training_dir):
        """End-to-end: load per-model fixture, run test_targets, verify outputs."""
        config = classify_pipeline_first.configs[0]
        config.data["step_class_set"]["steps"]["model"] = case.config_name

        ds = ClassifyAll(
            config,
            test_sets=classify_pipeline_first.extracts[0].target_features,
        )
        ds.model_file_names = {
            tgt: str(training_dir / f"model_{tgt}_{case.joblib_suffix}.joblib")
            for tgt in TARGETS
        }
        ds.read_models()
        ds.test_targets()

        for tgt in TARGETS:
            assert isinstance(ds.test_sets[tgt], pl.DataFrame)
            # TODO: update to actual value after data reduction
            assert ds.test_sets[tgt].shape[0] == 812
            assert ds.test_sets[tgt].shape[1] == 56

            assert isinstance(ds.contingency_tables[tgt], pl.DataFrame)
            # TODO: update to actual value after data reduction
            assert ds.contingency_tables[tgt].height == 812

        assert ds.contingency_tables["temp"].columns == [
            "k", "label", "predicted_label", "score",
        ]