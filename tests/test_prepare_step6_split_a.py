"""Unit tests for the ``SplitDataSetA`` class.

SplitDataSetA takes the ``target_features`` produced by ExtractDataSetA and
splits each target's frame into a training set and a test set, controlled
by ``test_set_fraction`` and ``k_fold`` config params.

Refactored from two ``unittest.TestCase`` classes into pytest classes
sharing the conftest ``build_prepare_pipeline()`` helper (with
``stop_after="extract"``). Per-target triplication collapses to
``for tgt in TARGETS:`` loops.
"""

import os

import polars as pl
import pytest

from aiqclib.prepare.step6_split_dataset.dataset_a import SplitDataSetA

from tests.conftest import TARGETS, build_prepare_pipeline


# ---------------------------------------------------------------------------
# Tests against config 001 (default)
# ---------------------------------------------------------------------------

class TestSplitDataSetA:
    """Tests for SplitDataSetA against config 001 (default selection)."""

    @pytest.fixture
    def pipeline(self, dataset_config_001, test_data_file):
        """Run the prepare pipeline through step5 (extract)."""
        return build_prepare_pipeline(
            dataset_config_001, test_data_file, stop_after="extract",
        )

    def test_step_name(self, pipeline):
        """step_name == 'split'."""
        ds = SplitDataSetA(pipeline.config)
        assert ds.step_name == "split"

    def test_output_file_names(self, pipeline):
        """Default per-target output paths for both train/ and test/ kinds."""
        ds = SplitDataSetA(pipeline.config)
        base = "/path/to/split_1/nrt_bo_001/split_folder_1"
        for kind in ("train", "test"):
            for tgt in TARGETS:
                assert (
                    str(ds.output_file_names[kind][tgt])
                    == f"{base}/{kind}_set_{tgt}.parquet"
                )

    def test_target_features_data(self, pipeline):
        """target_features lands on the instance with expected per-target shape."""
        ds = SplitDataSetA(pipeline.config, target_features=pipeline.extract.target_features)

        expected_rows = {"temp": 24, "psal": 36, "pres": 18}
        for tgt in TARGETS:
            assert isinstance(ds.target_features[tgt], pl.DataFrame)
            assert ds.target_features[tgt].shape[0] == expected_rows[tgt]
            assert ds.target_features[tgt].shape[1] == 58

    def test_split_features_data(self, pipeline):
        """process_targets produces (training_set, test_set) per target."""
        ds = SplitDataSetA(pipeline.config, target_features=pipeline.extract.target_features)
        ds.process_targets()

        expected_train_rows = {"temp": 22, "psal": 34, "pres": 18}
        expected_test_rows = {"temp": 2, "psal": 2, "pres": 0}
        for tgt in TARGETS:
            assert isinstance(ds.training_sets[tgt], pl.DataFrame)
            assert ds.training_sets[tgt].shape[0] == expected_train_rows[tgt]
            assert ds.training_sets[tgt].shape[1] == 57

            assert isinstance(ds.test_sets[tgt], pl.DataFrame)
            assert ds.test_sets[tgt].shape[0] == expected_test_rows[tgt]
            assert ds.test_sets[tgt].shape[1] == 56

    def test_default_test_set_fraction(self, pipeline):
        """test_set_fraction defaults to 0.1 when not set in config."""
        ds = SplitDataSetA(pipeline.config, target_features=pipeline.extract.target_features)
        ds.config.data["step_param_set"]["steps"]["split"]["test_set_fraction"] = None
        assert ds.get_test_set_fraction() == 0.1

    def test_default_k_fold(self, pipeline):
        """k_fold defaults to 10 when not set in config."""
        ds = SplitDataSetA(pipeline.config, target_features=pipeline.extract.target_features)
        ds.config.data["step_param_set"]["steps"]["split"]["k_fold"] = None
        assert ds.get_k_fold() == 10

    def test_write_training_sets(self, pipeline, test_output_dir):
        """write_training_sets produces a parquet per target."""
        ds = SplitDataSetA(pipeline.config, target_features=pipeline.extract.target_features)
        ds.process_targets()

        output_paths = {
            tgt: str(test_output_dir / f"test_train_set_{tgt}.parquet") for tgt in TARGETS
        }
        for tgt in TARGETS:
            ds.output_file_names["train"][tgt] = output_paths[tgt]

        ds.write_training_sets()

        for tgt in TARGETS:
            assert os.path.exists(output_paths[tgt])
            os.remove(output_paths[tgt])  # comment out to debug

    def test_write_empty_training_sets(self, pipeline):
        """write_training_sets with training_sets=None raises ValueError."""
        ds = SplitDataSetA(pipeline.config, target_features=pipeline.extract.target_features)
        ds.process_targets()
        ds.training_sets = None
        with pytest.raises(ValueError):
            ds.write_training_sets()

    def test_write_test_sets(self, pipeline, test_output_dir):
        """write_test_sets produces a parquet per target."""
        ds = SplitDataSetA(pipeline.config, target_features=pipeline.extract.target_features)
        ds.process_targets()

        output_paths = {
            tgt: str(test_output_dir / f"test_test_set_{tgt}.parquet") for tgt in TARGETS
        }
        for tgt in TARGETS:
            ds.output_file_names["test"][tgt] = output_paths[tgt]

        ds.write_test_sets()

        for tgt in TARGETS:
            assert os.path.exists(output_paths[tgt])
            os.remove(output_paths[tgt])  # comment out to debug

    def test_write_empty_test_sets(self, pipeline):
        """write_test_sets with test_sets=None raises ValueError."""
        ds = SplitDataSetA(pipeline.config, target_features=pipeline.extract.target_features)
        ds.process_targets()
        ds.test_sets = None
        with pytest.raises(ValueError):
            ds.write_test_sets()

    def test_write_data_sets(self, pipeline, test_output_dir):
        """write_data_sets produces both train and test parquets per target."""
        ds = SplitDataSetA(pipeline.config, target_features=pipeline.extract.target_features)
        ds.process_targets()

        output_paths = {}
        for kind in ("train", "test"):
            output_paths[kind] = {
                tgt: str(test_output_dir / f"test_{kind}_set_combined_{tgt}.parquet")
                for tgt in TARGETS
            }
            for tgt in TARGETS:
                ds.output_file_names[kind][tgt] = output_paths[kind][tgt]

        ds.write_data_sets()

        for kind in ("train", "test"):
            for tgt in TARGETS:
                assert os.path.exists(output_paths[kind][tgt])
                os.remove(output_paths[kind][tgt])  # comment out to debug


# ---------------------------------------------------------------------------
# Tests against config 003 (NegX5)
# ---------------------------------------------------------------------------

class TestSplitDataSetANegX5:
    """Tests for SplitDataSetA against config 003 (NegX5).

    The NegX5 variant produces much larger target_features (because negative
    rows were multiplied at step4). test_split_features_data here asserts
    that train+test row sums match the input total per target, rather than
    asserting specific train/test counts (since the split fraction varies).
    """

    @pytest.fixture
    def pipeline(self, dataset_config_003, test_data_file):
        return build_prepare_pipeline(
            dataset_config_003, test_data_file, stop_after="extract",
        )

    def test_target_features_data(self, pipeline):
        """NegX5 target_features have the multiplied-negative row counts."""
        ds = SplitDataSetA(pipeline.config, target_features=pipeline.extract.target_features)

        expected_rows = {"temp": 177, "psal": 249, "pres": 129}
        for tgt in TARGETS:
            assert isinstance(ds.target_features[tgt], pl.DataFrame)
            assert ds.target_features[tgt].shape[0] == expected_rows[tgt]
            assert ds.target_features[tgt].shape[1] == 58

    def test_split_features_data(self, pipeline):
        """train + test rows must sum to target_features rows per target.

        The exact split varies with config, but the conservation law holds.
        """
        ds = SplitDataSetA(pipeline.config, target_features=pipeline.extract.target_features)
        ds.process_targets()

        expected_totals = {"temp": 177, "psal": 249, "pres": 129}
        for tgt in TARGETS:
            assert isinstance(ds.training_sets[tgt], pl.DataFrame)
            assert isinstance(ds.test_sets[tgt], pl.DataFrame)
            assert ds.training_sets[tgt].shape[1] == 57
            assert ds.test_sets[tgt].shape[1] == 56
            assert (
                ds.training_sets[tgt].shape[0] + ds.test_sets[tgt].shape[0]
                == expected_totals[tgt]
            )

    def test_write_data_sets(self, pipeline, test_output_dir):
        """write_data_sets (NegX5 variant) produces both train and test parquets."""
        ds = SplitDataSetA(pipeline.config, target_features=pipeline.extract.target_features)
        ds.process_targets()

        output_paths = {}
        for kind in ("train", "test"):
            output_paths[kind] = {
                tgt: str(test_output_dir / f"test_{kind}_set_negx5_{tgt}.parquet")
                for tgt in TARGETS
            }
            for tgt in TARGETS:
                ds.output_file_names[kind][tgt] = output_paths[kind][tgt]

        ds.write_data_sets()

        for kind in ("train", "test"):
            for tgt in TARGETS:
                assert os.path.exists(output_paths[kind][tgt])
                os.remove(output_paths[kind][tgt])  # comment out to debug