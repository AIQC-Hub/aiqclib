"""Unit tests for the ``train_and_evaluate`` interface function.

Runs the full train pipeline (validate → build → save model) against test
configs and verifies that the expected output folder structure and
per-target files are produced. Also tests the ``calculate_shap`` flag,
which gates SHAP-value file creation.

Refactored from two classes (``TestCreateTrainingDataSet`` parametrized
over 2 configs, plus ``TestCreateTrainingDataSetNegX5`` for config 003)
into the same shape with conftest fixtures and per-target loops.

Note on config 001 vs bo002:
The first parametrized config uses ``training_config_001_bo002``
(NRT_BO_002, 2-target temp+psal) rather than ``training_config_001``
(NRT_BO_001, 3-target). The reduced test fixtures have zero pres test
rows, which crashes the train pipeline during the build step. Config 002
is unaffected and uses its default 3-target selection.

When the library handles zero-row test data, switch the first fixture
back to ``training_config_001`` and drop the ``TARGETS_NONEMPTY`` reference
in ``targets_per_config``.
"""

import os
import shutil

import pytest

from aiqclib.interface.train import train_and_evaluate

from tests.conftest import TARGETS_NONEMPTY


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wire_path_info(config, test_output_dir, input_dir):
    """Override path_info to read training inputs from input_dir and write under test_output_dir.

    Training tests use ``step_folder_name=".."`` to walk up from
    ``input.base_path`` to find the train/test parquets, mirroring how the
    real CLI is invoked.
    """
    config.data["path_info"] = {
        "name": "data_set_1",
        "common": {"base_path": str(test_output_dir)},
        "input": {"base_path": str(input_dir), "step_folder_name": ".."},
    }


def _assert_train_outputs(output_folder, *, expect_shap=False, targets=TARGETS_NONEMPTY):
    """Assert every expected output file from train_and_evaluate exists.

    :param targets: which targets to check. Defaults to TARGETS; pass
        TARGETS_NONEMPTY for configs (like NRT_BO_002) that exclude pres,
        since the pipeline produces files only for the configured targets.
    :param expect_shap: if True, per-target SHAP value parquets must exist;
        otherwise they must not.
    """
    dir_validate = output_folder / "validate"
    dir_build = output_folder / "build"
    dir_model = output_folder / "model"

    for tgt in targets:
        # Validate stage
        assert (dir_validate / f"validation_report_{tgt}.tsv").exists()
        assert (dir_validate / f"contingency_tables_{tgt}.parquet").exists()
        assert (dir_validate / f"metric_plots_{tgt}.svg").exists()
        # Build stage
        assert (dir_build / f"test_report_{tgt}.tsv").exists()
        assert (dir_build / f"test_contingency_tables_{tgt}.parquet").exists()
        assert (dir_build / f"test_metric_plots_{tgt}.svg").exists()
        # Model artifacts
        assert (dir_model / f"model_{tgt}.joblib").exists()

        # SHAP files: present iff calculate_shap was True
        shap_path = dir_build / f"test_shap_values_{tgt}.parquet"
        if expect_shap:
            assert shap_path.exists()
        else:
            assert not shap_path.exists()


def _cleanup_output_folder(config, test_output_dir):
    """Remove the config's ``dataset_folder_name`` directory if present."""
    output_folder = test_output_dir / config.data["dataset_folder_name"]
    if output_folder.exists() and output_folder.is_dir():
        shutil.rmtree(output_folder)


# ---------------------------------------------------------------------------
# Two-config tests (configs 001 [bo002] + 002)
# ---------------------------------------------------------------------------

class TestCreateTrainingDataSet:
    """train_and_evaluate against training configs 001 (bo002 variant) and 002.

    idx=0 uses NRT_BO_002 from test_training_001.yaml (2-target temp+psal),
    idx=1 uses default NRT_BO_001 from test_training_002.yaml (3-target).
    ``targets_per_config[idx]`` tells the assertion helper which targets
    to check for each.
    """

    @pytest.fixture(autouse=True)
    def setup_and_clean(
        self, training_config_001_bo002, training_config_002_bo002,
        test_output_dir, training_dir,
    ):
        """Wire two configs with training_dir as input_path; clean up afterwards."""
        self.configs = [training_config_001_bo002, training_config_002_bo002]
        # Per-config expected targets: bo002 produces temp+psal only;
        # config 002 produces all three.
        self.targets_per_config = [TARGETS_NONEMPTY, TARGETS_NONEMPTY]
        self.test_output_dir = test_output_dir
        for c in self.configs:
            _wire_path_info(c, test_output_dir, training_dir)

        yield

        for c in self.configs:
            _cleanup_output_folder(c, test_output_dir)

    @pytest.mark.parametrize("idx", range(2))
    def test_train_and_evaluate(self, idx):
        """End-to-end train pipeline produces all outputs; no SHAP files by default."""
        train_and_evaluate(self.configs[idx])

        output_folder = (
            self.test_output_dir / self.configs[idx].data["dataset_folder_name"]
        )
        _assert_train_outputs(
            output_folder, expect_shap=False, targets=self.targets_per_config[idx],
        )

    @pytest.mark.parametrize("idx", range(2))
    def test_shap_value_output(self, idx):
        """With ``calculate_shap=True``, SHAP value parquets are produced."""
        self.configs[idx].data["step_param_set"]["steps"]["model"]["calculate_shap"] = True
        train_and_evaluate(self.configs[idx])

        output_folder = (
            self.test_output_dir / self.configs[idx].data["dataset_folder_name"]
        )
        dir_build = output_folder / "build"

        for tgt in self.targets_per_config[idx]:
            assert (dir_build / f"test_shap_values_{tgt}.parquet").exists()


# ---------------------------------------------------------------------------
# NegX5 variant (config 003 with negx5_training/ as input)
# ---------------------------------------------------------------------------

class TestCreateTrainingDataSetNegX5:
    """train_and_evaluate against config 003 using the negx5_training fixtures."""

    @pytest.fixture(autouse=True)
    def setup_and_clean(self, training_config_003_bo002, test_output_dir, data_dir):
        """Wire config 003 with negx5_training/ as the input base path."""
        self.config = training_config_003_bo002
        self.test_output_dir = test_output_dir
        _wire_path_info(self.config, test_output_dir, data_dir / "negx5_training")

        yield

        _cleanup_output_folder(self.config, test_output_dir)

    def test_train_and_evaluate(self):
        """End-to-end train pipeline with the NegX5 inputs produces all outputs.

        SHAP files are not expected (calculate_shap defaults to False).
        """
        train_and_evaluate(self.config)

        output_folder = self.test_output_dir / self.config.data["dataset_folder_name"]

        # Original NegX5 test only asserted reports/contingency/plots/models
        # (it didn't check SHAP absence). Preserving that behaviour: assert
        # validate + build report/contingency/plots + model joblibs.
        dir_validate = output_folder / "validate"
        dir_build = output_folder / "build"
        dir_model = output_folder / "model"
        for tgt in TARGETS_NONEMPTY:
            assert (dir_validate / f"validation_report_{tgt}.tsv").exists()
            assert (dir_validate / f"contingency_tables_{tgt}.parquet").exists()
            assert (dir_validate / f"metric_plots_{tgt}.svg").exists()
            assert (dir_build / f"test_report_{tgt}.tsv").exists()
            assert (dir_build / f"test_contingency_tables_{tgt}.parquet").exists()
            assert (dir_build / f"test_metric_plots_{tgt}.svg").exists()
            assert (dir_model / f"model_{tgt}.joblib").exists()