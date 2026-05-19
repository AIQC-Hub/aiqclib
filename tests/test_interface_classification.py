"""Unit tests for the ``classify_dataset`` interface function.

Runs the full classify pipeline (summary → select → locate → extract →
classify → concat) and verifies that the expected output folder structure
and per-target files are produced. Also tests the ``calculate_shap`` flag
and the NegX5 variant (which uses a different model directory).

Refactored from two classes (``TestClassifyDataSet`` parametrized over 2
configs, plus ``TestClassifyDataSetNegX5`` for config 001 with the
``negx5_model`` model directory) into the same shape with conftest
fixtures and per-target loops.
"""

import shutil

import pytest

from aiqclib.interface.classify import classify_dataset

from tests.conftest import TARGETS_NONEMPTY


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wire_path_info(config, test_output_dir, input_dir, data_dir, *, model_subfolder):
    """Override path_info for classify tests.

    ``model_subfolder`` distinguishes the default tests ("training") from
    the NegX5 variant ("negx5_model"), which loads a different set of
    serialised models.
    """
    config.data["input_file_name"] = "nrt_cora_bo_test.parquet"
    config.data["path_info"] = {
        "name": "data_set_1",
        "common": {"base_path": str(test_output_dir)},
        "input": {"base_path": str(input_dir), "step_folder_name": ""},
        "model": {"base_path": str(data_dir), "step_folder_name": model_subfolder},
        "concat": {"step_folder_name": "classify"},
    }


def _assert_classify_outputs(output_folder, *, expect_shap=False):
    """Assert every expected output file from classify_dataset exists.

    If ``expect_shap`` is True, the per-target SHAP value parquets must
    also exist; otherwise they must not.
    """
    dir_summary = output_folder / "summary"
    dir_select = output_folder / "select"
    dir_locate = output_folder / "locate"
    dir_extract = output_folder / "extract"
    dir_classify = output_folder / "classify"

    assert (dir_summary / "summary_stats_classify.tsv").exists()
    assert (dir_select / "selected_profiles_classify.parquet").exists()

    for tgt in TARGETS_NONEMPTY:
        assert (dir_locate / f"selected_rows_classify_{tgt}.parquet").exists()
        assert (dir_extract / f"extracted_features_classify_{tgt}.parquet").exists()
        assert (dir_classify / f"classify_prediction_{tgt}.parquet").exists()
        assert (dir_classify / f"classify_report_{tgt}.tsv").exists()
        assert (dir_classify / f"classify_contingency_tables_{tgt}.parquet").exists()
        assert (dir_classify / f"classify_metric_plots_{tgt}.svg").exists()

        shap_path = dir_classify / f"classify_shap_values_{tgt}.parquet"
        if expect_shap:
            assert shap_path.exists()
        else:
            assert not shap_path.exists()

    # Concatenated predictions file (not per-target).
    assert (dir_classify / "predictions.parquet").exists()


def _cleanup_output_folder(config, test_output_dir):
    """Remove the config's ``dataset_folder_name`` directory if present."""
    output_folder = test_output_dir / config.data["dataset_folder_name"]
    if output_folder.exists() and output_folder.is_dir():
        shutil.rmtree(output_folder)


# ---------------------------------------------------------------------------
# Default classification tests (model dir = "training")
# ---------------------------------------------------------------------------


class TestClassifyDataSet:
    """classify_dataset against classify configs 001 and 002, default model dir."""

    @pytest.fixture(autouse=True)
    def setup_and_cleanup(
        self,
        classify_config_001,
        classify_config_002,
        test_output_dir,
        input_dir,
        data_dir,
    ):
        """Wire two configs with model_subfolder='training'; clean up afterwards."""
        self.configs = [classify_config_001, classify_config_002]
        self.test_output_dir = test_output_dir
        for c in self.configs:
            _wire_path_info(
                c,
                test_output_dir,
                input_dir,
                data_dir,
                model_subfolder="training",
            )

        yield

        for c in self.configs:
            _cleanup_output_folder(c, test_output_dir)

    @pytest.mark.parametrize("idx", range(2))
    def test_classify_data_set(self, idx):
        """End-to-end classify pipeline produces all outputs; no SHAP by default."""
        classify_dataset(self.configs[idx])

        output_folder = (
            self.test_output_dir / self.configs[idx].data["dataset_folder_name"]
        )
        _assert_classify_outputs(output_folder, expect_shap=False)

    @pytest.mark.parametrize("idx", range(2))
    def test_shap_value_output(self, idx):
        """With ``calculate_shap=True``, SHAP value parquets are produced."""
        self.configs[idx].data["step_param_set"]["steps"]["model"]["calculate_shap"] = (
            True
        )
        classify_dataset(self.configs[idx])

        output_folder = (
            self.test_output_dir / self.configs[idx].data["dataset_folder_name"]
        )
        dir_classify = output_folder / "classify"

        for tgt in TARGETS_NONEMPTY:
            assert (dir_classify / f"classify_shap_values_{tgt}.parquet").exists()


# ---------------------------------------------------------------------------
# NegX5 variant (config 001 with negx5_model/ as the model directory)
# ---------------------------------------------------------------------------


class TestClassifyDataSetNegX5:
    """classify_dataset against config 001 but with negx5_model/ as model dir.

    The negx5_model directory contains XGBoost models trained on the
    negx5_training fixtures (regenerated by scripts/regenerate_test_models.py).
    This test verifies that classify_dataset uses whatever models are
    pointed at by path_info.model.
    """

    @pytest.fixture(autouse=True)
    def setup_and_cleanup(
        self,
        classify_config_001,
        test_output_dir,
        input_dir,
        data_dir,
    ):
        """Wire config 001 with model_subfolder='negx5_model'."""
        self.config = classify_config_001
        self.test_output_dir = test_output_dir
        _wire_path_info(
            self.config,
            test_output_dir,
            input_dir,
            data_dir,
            model_subfolder="negx5_model",
        )

        yield

        _cleanup_output_folder(self.config, test_output_dir)

    def test_classify_data_set(self):
        """End-to-end classify pipeline with the negx5_model models produces all outputs.

        Original NegX5 test didn't assert SHAP absence (no calculate_shap
        setup); preserving that — _assert_classify_outputs default
        ``expect_shap=False`` checks the SHAP files are absent.
        """
        classify_dataset(self.config)

        output_folder = self.test_output_dir / self.config.data["dataset_folder_name"]
        _assert_classify_outputs(output_folder, expect_shap=False)
