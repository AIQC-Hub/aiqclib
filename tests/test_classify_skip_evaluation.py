"""Tests for the label-free classification path (``skip_evaluation``).

Covers the feature that lets the Classification module run on datasets with no
QC flag / label column:

- Config resolution: :meth:`ConfigBase.is_flag_missing` and
  :meth:`ConfigBase.get_skip_evaluation` (per-target derivation + step override).
- Schema: the classification ``target_sets`` no longer requires ``flag`` while
  the prepare/training schemas still do.
- step4 (``LocateDataSetAll``): the skip path keeps every row with null
  ``flag``/``label`` and does not need a QC column.
- step6 (``ClassifyAll``): the skip path produces predictions but no report /
  model-scores, and the writers + metric plots tolerate the missing outputs.
"""

import os

import polars as pl
import pytest
import yaml

from aiqclib.classify.step4_select_rows.dataset_all import LocateDataSetAll
from aiqclib.classify.step6_classify_dataset.dataset_all import ClassifyAll
from aiqclib.classify.step7_concat_datasets.dataset_all import ConcatDataSetAll
from aiqclib.common.base.config_base import ConfigBase
from aiqclib.common.config.yaml_schema import (
    get_classification_config_schema,
    get_data_set_config_schema,
    get_training_config_schema,
)

from tests.conftest import TARGETS_NONEMPTY, run_classify_prepare_pipeline


def _variables_required(schema_str: str) -> list:
    """Drill into a schema's target_sets → variables item ``required`` list."""
    schema = yaml.safe_load(schema_str)
    variables = schema["properties"]["target_sets"]["items"]["properties"]["variables"]
    return variables["items"]["required"]


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------


class TestSkipEvaluationConfig:
    """Unit tests for the flag-missing / skip_evaluation config helpers."""

    @pytest.mark.parametrize(
        "target_value, expected",
        [
            ({"name": "temp", "flag": "temp_qc"}, False),
            ({"name": "temp", "flag": ""}, True),
            ({"name": "temp", "flag": "   "}, True),
            ({"name": "temp", "flag": None}, True),
            ({"name": "temp"}, True),
        ],
    )
    def test_is_flag_missing(self, target_value, expected):
        """A flag is 'missing' when absent, None, or empty/whitespace."""
        assert ConfigBase.is_flag_missing(target_value) is expected

    def test_get_skip_evaluation_derived_per_target(self, classify_config_001):
        """With no override, skip is derived per target from its flag."""
        variables = classify_config_001.data["target_set"]["variables"]
        # temp keeps its flag; psal loses it.
        by_name = {v["name"]: v for v in variables}
        by_name["psal"].pop("flag")

        assert classify_config_001.get_skip_evaluation("temp") is False
        assert classify_config_001.get_skip_evaluation("psal") is True

    def test_get_skip_evaluation_step_override(self, classify_config_001):
        """An explicit step-level override wins over per-target derivation."""
        model_params = classify_config_001.data["step_param_set"]["steps"]["model"]

        # temp has a valid flag (derived False), but the override forces True.
        model_params["skip_evaluation"] = True
        assert classify_config_001.get_skip_evaluation("temp") is True

        # Drop temp's flag (derived True), but the override forces False.
        by_name = {
            v["name"]: v for v in classify_config_001.data["target_set"]["variables"]
        }
        by_name["temp"].pop("flag")
        model_params["skip_evaluation"] = False
        assert classify_config_001.get_skip_evaluation("temp") is False


# ---------------------------------------------------------------------------
# Schema relaxation (classify only)
# ---------------------------------------------------------------------------


class TestSkipEvaluationSchema:
    """The relaxation must apply to the classification schema only."""

    def test_classify_schema_flag_optional(self):
        """Classification variables require only ``name``; flag is nullable."""
        assert _variables_required(get_classification_config_schema()) == ["name"]

        schema = yaml.safe_load(get_classification_config_schema())
        flag_type = schema["properties"]["target_sets"]["items"]["properties"][
            "variables"
        ]["items"]["properties"]["flag"]["type"]
        assert "null" in flag_type

    def test_prepare_and_training_schema_flag_required(self):
        """Prepare/training schemas stay strict about the flag fields."""
        expected = ["name", "flag", "pos_flag_values", "neg_flag_values"]
        assert _variables_required(get_data_set_config_schema()) == expected
        assert _variables_required(get_training_config_schema()) == expected


# ---------------------------------------------------------------------------
# step4: label-free row selection
# ---------------------------------------------------------------------------


class TestSkipEvaluationStep4:
    """LocateDataSetAll keeps all rows and null labels on the skip path."""

    def test_skip_keeps_all_rows_with_null_labels(
        self, classify_config_001, classify_input_001, classify_select_001
    ):
        """Dropping temp's flag → temp is label-free; psal is still labelled."""
        by_name = {
            v["name"]: v for v in classify_config_001.data["target_set"]["variables"]
        }
        by_name["temp"].pop("flag")

        ds = LocateDataSetAll(
            classify_config_001,
            input_data=classify_input_001.input_data,
            selected_profiles=classify_select_001.selected_profiles,
        )
        ds.process_targets()

        # temp: label-free — all rows, flag and label fully null.
        temp = ds.selected_rows["temp"]
        assert temp.shape == (2456, 9)
        assert temp["flag"].null_count() == 2456
        assert temp["label"].null_count() == 2456

        # psal: still labelled — labels present (0/1), no nulls.
        psal = ds.selected_rows["psal"]
        assert psal["label"].null_count() == 0
        assert set(psal["label"].unique().to_list()) <= {0, 1}

    def test_skip_works_without_qc_column(
        self, classify_config_001, classify_input_001, classify_select_001
    ):
        """The skip path does not reference the QC column, so it can be absent."""
        by_name = {
            v["name"]: v for v in classify_config_001.data["target_set"]["variables"]
        }
        by_name["temp"].pop("flag")

        # Physically remove the QC column to prove it is not required.
        input_no_qc = classify_input_001.input_data.drop("temp_qc")

        ds = LocateDataSetAll(
            classify_config_001,
            input_data=input_no_qc,
            selected_profiles=classify_select_001.selected_profiles,
        )
        ds.locate_target_rows("temp", by_name["temp"])

        assert ds.selected_rows["temp"].shape == (2456, 9)


# ---------------------------------------------------------------------------
# step6/step7: label-free classification (end-to-end, XGBoost defaults)
# ---------------------------------------------------------------------------


@pytest.fixture
def skip_pipeline(test_data_file, classify_yaml_001):
    """Prepare pipeline for classify_001 with skip_evaluation forced on.

    The override is injected *before* the pipeline runs so step4 also takes the
    label-free path and null labels propagate through to the extracted test set
    — the true end-to-end skip scenario.
    """

    def _enable_skip(config):
        config.data["step_param_set"]["steps"]["model"]["skip_evaluation"] = True

    configs, extracts = run_classify_prepare_pipeline(
        [classify_yaml_001], test_data_file, mutate_config=_enable_skip
    )
    return configs[0], extracts[0]


def _make_classify_ds(config, extract, training_dir):
    """Build a ClassifyAll with default XGBoost models loaded."""
    ds = ClassifyAll(config, test_sets=extract.target_features)
    ds.model_file_names = {
        tgt: str(training_dir / f"model_{tgt}.joblib") for tgt in TARGETS_NONEMPTY
    }
    ds.read_models()
    return ds


class TestSkipEvaluationStep6:
    """ClassifyAll predicts without labels and skips evaluation outputs."""

    def test_predicts_without_evaluation(self, skip_pipeline, training_dir):
        """Predictions are produced; report and model-scores are skipped."""
        config, extract = skip_pipeline
        ds = _make_classify_ds(config, extract, training_dir)
        ds.test_targets()

        for tgt in TARGETS_NONEMPTY:
            # Predictions exist with a null label column.
            preds = ds.predictions[tgt]
            assert isinstance(preds, pl.DataFrame)
            assert preds.shape[0] == 2456
            assert preds["label"].null_count() == 2456
            assert {"predicted_label", "score"} <= set(preds.columns)

            # No evaluation artefacts.
            assert ds.reports[tgt] is None
            assert ds.model_scores[tgt] is None

    def test_writers_and_plots_tolerate_skip(
        self, skip_pipeline, training_dir, test_output_dir
    ):
        """Writers/plots run without error and emit only predictions."""
        config, extract = skip_pipeline
        ds = _make_classify_ds(config, extract, training_dir)

        pred_paths = {
            tgt: str(test_output_dir / f"skip_pred_{tgt}.parquet")
            for tgt in TARGETS_NONEMPTY
        }
        report_paths = {
            tgt: str(test_output_dir / f"skip_report_{tgt}.tsv")
            for tgt in TARGETS_NONEMPTY
        }
        score_paths = {
            tgt: str(test_output_dir / f"skip_scores_{tgt}.parquet")
            for tgt in TARGETS_NONEMPTY
        }
        plot_paths = {
            tgt: str(test_output_dir / f"skip_plot_{tgt}.svg")
            for tgt in TARGETS_NONEMPTY
        }
        ds.output_file_names["prediction"] = pred_paths
        ds.output_file_names["report"] = report_paths
        ds.output_file_names["model_scores"] = score_paths
        ds.output_file_names["metric_plot"] = plot_paths

        ds.test_targets()

        # None of these should raise even though there is nothing to evaluate.
        ds.write_reports()
        ds.write_model_scores()
        ds.create_metric_plots()
        ds.write_predictions()

        for tgt in TARGETS_NONEMPTY:
            # Predictions written; evaluation artefacts skipped entirely.
            assert os.path.exists(pred_paths[tgt])
            assert not os.path.exists(report_paths[tgt])
            assert not os.path.exists(score_paths[tgt])
            assert not os.path.exists(plot_paths[tgt])
            os.remove(pred_paths[tgt])


class TestSkipEvaluationStep7:
    """step7 concat aligns/joins predictions even with null label columns."""

    def test_merge_predictions_with_null_labels(
        self, skip_pipeline, training_dir, classify_input_001
    ):
        """merge_predictions keeps every input row; per-target labels are null."""
        config, extract = skip_pipeline
        ds_classify = _make_classify_ds(config, extract, training_dir)
        ds_classify.test_targets()

        ds_concat = ConcatDataSetAll(
            config,
            input_data=classify_input_001.input_data,
            predictions=ds_classify.predictions,
        )
        ds_concat.merge_predictions()

        merged = ds_concat.merged_predictions
        assert merged.shape[0] == 2456
        for tgt in TARGETS_NONEMPTY:
            assert merged[f"{tgt}_label"].null_count() == 2456
            assert f"{tgt}_predicted" in merged.columns
            assert f"{tgt}_score" in merged.columns
