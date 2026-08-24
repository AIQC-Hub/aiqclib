"""End-to-end tests for the profile-level prepare pipeline.

Runs ``create_training_dataset`` with the profile-level step classes
(LocateDataSetProfile → ExtractDataSetProfile → SplitDataSetProfile) for
both label modes, and chains the binary output into ``train_and_evaluate``
with the existing XGBoost classifier to prove the training stage consumes
profile-level frames (no ``observation_no`` column) unchanged.
"""

import shutil

import polars as pl
import pytest

from aiqclib.common.config.training_config import TrainingConfig
from aiqclib.interface.prepare import create_training_dataset
from aiqclib.interface.train import train_and_evaluate

from tests.conftest import CONFIG_DIR, TARGETS, TARGETS_NONEMPTY

PROFILE_FOLDER = "nrt_bo_profile_e2e"


def _wire_prepare(config, test_output_dir, input_dir):
    """Redirect the prepare config's output under test_output_dir."""
    config.data["input_file_name"] = "nrt_cora_bo_test.parquet"
    config.data["dataset_folder_name"] = PROFILE_FOLDER
    config.data["path_info"] = {
        "name": "data_set_1",
        "common": {"base_path": str(test_output_dir)},
        "input": {"base_path": str(input_dir), "step_folder_name": ""},
    }


def _load_train_config(test_output_dir):
    """Training config (temp+psal, XGBoost) reading the profile split output."""
    config = TrainingConfig(str(CONFIG_DIR / "test_training_001.yaml"))
    config.select("NRT_BO_002")
    config.data["dataset_folder_name"] = PROFILE_FOLDER
    config.data["path_info"] = {
        "name": "data_set_1",
        "common": {"base_path": str(test_output_dir)},
        "input": {"base_path": str(test_output_dir), "step_folder_name": "split"},
    }
    return config


@pytest.fixture
def cleanup_profile_folder(test_output_dir):
    yield
    output_folder = test_output_dir / PROFILE_FOLDER
    if output_folder.exists() and output_folder.is_dir():
        shutil.rmtree(output_folder)


class TestCreateTrainingDataSetProfile:
    """create_training_dataset with the profile-level step classes."""

    def test_binary_prepare_and_train(
        self,
        dataset_config_profile,
        test_output_dir,
        input_dir,
        cleanup_profile_folder,
    ):
        """Binary profile labels: prepare produces per-profile sets, XGB trains on them."""
        _wire_prepare(dataset_config_profile, test_output_dir, input_dir)
        create_training_dataset(dataset_config_profile)

        output_folder = test_output_dir / PROFILE_FOLDER
        for tgt in TARGETS:
            locate_file = output_folder / "locate" / f"selected_rows_{tgt}.parquet"
            observation_file = (
                output_folder / "locate" / f"selected_observation_rows_{tgt}.parquet"
            )
            train_file = output_folder / "split" / f"train_set_{tgt}.parquet"
            test_file = output_folder / "split" / f"test_set_{tgt}.parquet"
            for path in (locate_file, observation_file, train_file, test_file):
                assert path.exists(), path

            train_set = pl.read_parquet(train_file)
            assert "observation_no" not in train_set.columns
            assert set(train_set["label"].unique().to_list()) <= {0, 1}

        # The training stage consumes the profile-level frames unchanged.
        train_config = _load_train_config(test_output_dir)
        train_and_evaluate(train_config)

        for tgt in TARGETS_NONEMPTY:
            assert (output_folder / "model" / f"model_{tgt}.joblib").exists()
            assert (
                output_folder / "validate" / f"validation_report_{tgt}.tsv"
            ).exists()
            assert (output_folder / "build" / f"test_report_{tgt}.tsv").exists()

    def test_proportion_prepare(
        self,
        dataset_config_profile_proportion,
        test_output_dir,
        input_dir,
        cleanup_profile_folder,
    ):
        """Proportion labels flow through prepare as floats in [0, 1]."""
        _wire_prepare(dataset_config_profile_proportion, test_output_dir, input_dir)
        create_training_dataset(dataset_config_profile_proportion)

        output_folder = test_output_dir / PROFILE_FOLDER
        for tgt in TARGETS:
            train_set = pl.read_parquet(
                output_folder / "split" / f"train_set_{tgt}.parquet"
            )
            assert train_set["label"].dtype == pl.Float64
            assert (train_set["label"] >= 0.0).all()
            assert (train_set["label"] <= 1.0).all()
