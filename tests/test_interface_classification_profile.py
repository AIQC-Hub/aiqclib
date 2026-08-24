"""End-to-end tests for the profile-level classification pipeline.

Each test chains the real stages: profile-level prepare
(``create_training_dataset``) → train (``train_and_evaluate``) → classify
(``classify_dataset``) with the profile-level classify step classes, and
verifies the one-row-per-profile predictions output. Covers the binary /
classifier path, the proportion / regressor path, and a label-free target.
"""

import shutil

import polars as pl
import pytest

from aiqclib.common.config.training_config import TrainingConfig
from aiqclib.interface.classify import classify_dataset
from aiqclib.interface.prepare import create_training_dataset
from aiqclib.interface.train import train_and_evaluate

from tests.conftest import (
    CONFIG_DIR,
    TARGETS_NONEMPTY,
    _make_profile_dataset_config,
)

PROFILE_FOLDER = "nrt_bo_profile_classify_e2e"


def _run_prepare_and_train(test_output_dir, input_dir, *, label_mode, model_class):
    """Run profile prepare + train, leaving models under PROFILE_FOLDER/model."""
    prepare_config = _make_profile_dataset_config(label_mode)
    prepare_config.data["input_file_name"] = "nrt_cora_bo_test.parquet"
    prepare_config.data["dataset_folder_name"] = PROFILE_FOLDER
    prepare_config.data["path_info"] = {
        "name": "data_set_1",
        "common": {"base_path": str(test_output_dir)},
        "input": {"base_path": str(input_dir), "step_folder_name": ""},
    }
    create_training_dataset(prepare_config)

    train_config = TrainingConfig(str(CONFIG_DIR / "test_training_001.yaml"))
    train_config.select("NRT_BO_002")
    train_config.data["dataset_folder_name"] = PROFILE_FOLDER
    train_config.data["path_info"] = {
        "name": "data_set_1",
        "common": {"base_path": str(test_output_dir)},
        "input": {"base_path": str(test_output_dir), "step_folder_name": "split"},
    }
    train_config.data["step_class_set"]["steps"]["model"] = model_class
    train_and_evaluate(train_config)


def _wire_classify(config, test_output_dir, input_dir, model_class):
    """Point a profile classify config at the trained models and test output."""
    config.data["input_file_name"] = "nrt_cora_bo_test.parquet"
    config.data["dataset_folder_name"] = PROFILE_FOLDER
    config.data["path_info"] = {
        "name": "data_set_1",
        "common": {"base_path": str(test_output_dir)},
        "input": {"base_path": str(input_dir), "step_folder_name": ""},
        "model": {
            "base_path": str(test_output_dir / PROFILE_FOLDER),
            "step_folder_name": "model",
        },
        "concat": {"step_folder_name": "classify"},
    }
    config.data["step_class_set"]["steps"]["model"] = model_class


@pytest.fixture
def cleanup_profile_folder(test_output_dir):
    yield
    output_folder = test_output_dir / PROFILE_FOLDER
    if output_folder.exists() and output_folder.is_dir():
        shutil.rmtree(output_folder)


def _assert_profile_predictions(test_output_dir):
    """Check the one-row-per-profile predictions output."""
    predictions_file = (
        test_output_dir / PROFILE_FOLDER / "classify" / "predictions_profile.parquet"
    )
    assert predictions_file.exists()
    predictions = pl.read_parquet(predictions_file)
    assert "observation_no" not in predictions.columns
    assert (
        predictions.unique(subset=["platform_code", "profile_no"]).height
        == predictions.height
    )
    for tgt in TARGETS_NONEMPTY:
        for suffix in ("label", "predicted", "score"):
            assert f"{tgt}_{suffix}" in predictions.columns
        scores = predictions[f"{tgt}_score"].drop_nulls()
        assert (scores >= 0.0).all()
        assert (scores <= 1.0).all()
    return predictions


class TestClassifyDataSetProfile:
    """classify_dataset with the profile-level step classes."""

    def test_binary_classifier(
        self,
        classify_config_profile,
        test_output_dir,
        input_dir,
        cleanup_profile_folder,
    ):
        """Binary profile labels classified with the XGBoost classifier."""
        _run_prepare_and_train(
            test_output_dir, input_dir, label_mode="binary", model_class="XGBoost"
        )
        _wire_classify(classify_config_profile, test_output_dir, input_dir, "XGBoost")
        classify_dataset(classify_config_profile)

        output_folder = test_output_dir / PROFILE_FOLDER
        for tgt in TARGETS_NONEMPTY:
            assert (
                output_folder / "locate" / f"selected_rows_classify_{tgt}.parquet"
            ).exists()
            assert (
                output_folder / "extract" / f"extracted_features_classify_{tgt}.parquet"
            ).exists()
            assert (
                output_folder / "classify" / f"classify_prediction_{tgt}.parquet"
            ).exists()
        _assert_profile_predictions(test_output_dir)

    def test_proportion_regressor(
        self,
        classify_config_profile_proportion,
        test_output_dir,
        input_dir,
        cleanup_profile_folder,
    ):
        """Proportion labels classified with the XGBoost regressor."""
        _run_prepare_and_train(
            test_output_dir,
            input_dir,
            label_mode="proportion",
            model_class="XGBoostRegressor",
        )
        _wire_classify(
            classify_config_profile_proportion,
            test_output_dir,
            input_dir,
            "XGBoostRegressor",
        )
        classify_dataset(classify_config_profile_proportion)

        predictions = _assert_profile_predictions(test_output_dir)
        for tgt in TARGETS_NONEMPTY:
            assert predictions[f"{tgt}_label"].dtype == pl.Float64

    def test_label_free_target(
        self,
        classify_config_profile,
        test_output_dir,
        input_dir,
        cleanup_profile_folder,
    ):
        """A target without a QC flag is classified label-free (null labels)."""
        _run_prepare_and_train(
            test_output_dir, input_dir, label_mode="binary", model_class="XGBoost"
        )
        for variable in classify_config_profile.data["target_set"]["variables"]:
            if variable["name"] == "psal":
                variable["flag"] = None
        _wire_classify(classify_config_profile, test_output_dir, input_dir, "XGBoost")
        classify_dataset(classify_config_profile)

        predictions = _assert_profile_predictions(test_output_dir)
        # Label-free target: every profile predicted, labels all null
        assert predictions["psal_label"].null_count() == predictions.height
        assert predictions["psal_score"].null_count() == 0
        # Labelled target unaffected
        assert predictions["temp_label"].null_count() < predictions.height
