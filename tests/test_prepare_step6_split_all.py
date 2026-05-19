"""Unit tests for the ``SplitDataSetAll`` class.

SplitDataSetAll is the "select-all" variant of SplitDataSetA — it takes the
``target_features`` produced by ExtractDataSetAll (which keeps every input
row per target) and splits each target's frame into train/test sets.

Refactored from a single ``unittest.TestCase`` class. Uses
``build_prepare_pipeline(dataset_config_005, ..., stop_after="extract")``
to get the upstream pipeline output, then exercises SplitDataSetAll on top.
Per-target triplication collapses to ``for tgt in TARGETS:`` loops.
"""

import os

import polars as pl
import pytest

from aiqclib.prepare.step6_split_dataset.dataset_all import SplitDataSetAll

from tests.conftest import TARGETS, build_prepare_pipeline


class TestSplitDataSetAll:
    """Tests for SplitDataSetAll's train/test split and file output."""

    @pytest.fixture
    def pipeline(self, dataset_config_005, test_data_file):
        """Run the prepare pipeline through step5 (extract) for the select-all config."""
        return build_prepare_pipeline(
            dataset_config_005,
            test_data_file,
            stop_after="extract",
        )

    def test_step_name(self, pipeline):
        """step_name == 'split'."""
        ds = SplitDataSetAll(pipeline.config)
        assert ds.step_name == "split"

    def test_target_features_data(self, pipeline):
        """target_features arrives with the expected per-target shape (full input width)."""
        ds = SplitDataSetAll(
            pipeline.config,
            target_features=pipeline.extract.target_features,
        )

        for tgt in TARGETS:
            assert isinstance(ds.target_features[tgt], pl.DataFrame)
            assert ds.target_features[tgt].shape[0] == 3267
            assert ds.target_features[tgt].shape[1] == 58

    def test_split_features_data(self, pipeline):
        """process_targets produces non-empty (training_set, test_set) per target."""
        ds = SplitDataSetAll(
            pipeline.config,
            target_features=pipeline.extract.target_features,
        )
        ds.process_targets()

        expected_train_rows = {"temp": 2941, "psal": 2942, "pres": 2942}
        expected_test_rows = {"temp": 326, "psal": 325, "pres": 325}
        for tgt in TARGETS:
            assert isinstance(ds.training_sets[tgt], pl.DataFrame)
            assert ds.training_sets[tgt].shape[0] == expected_train_rows[tgt]
            assert ds.training_sets[tgt].shape[1] == 57

            assert isinstance(ds.test_sets[tgt], pl.DataFrame)
            assert ds.test_sets[tgt].shape[0] == expected_test_rows[tgt]
            assert ds.test_sets[tgt].shape[1] == 56

    def test_write_training_sets(self, pipeline, test_output_dir):
        """write_training_sets produces a parquet per target."""
        ds = SplitDataSetAll(
            pipeline.config,
            target_features=pipeline.extract.target_features,
        )
        ds.process_targets()

        output_paths = {
            tgt: str(test_output_dir / f"test_train_set_all_{tgt}.parquet")
            for tgt in TARGETS
        }
        for tgt in TARGETS:
            ds.output_file_names["train"][tgt] = output_paths[tgt]

        ds.write_training_sets()

        for tgt in TARGETS:
            assert os.path.exists(output_paths[tgt])
            os.remove(output_paths[tgt])  # comment out to debug

    def test_write_test_sets(self, pipeline, test_output_dir):
        """write_test_sets produces a parquet per target."""
        ds = SplitDataSetAll(
            pipeline.config,
            target_features=pipeline.extract.target_features,
        )
        ds.process_targets()

        output_paths = {
            tgt: str(test_output_dir / f"test_test_set_all_{tgt}.parquet")
            for tgt in TARGETS
        }
        for tgt in TARGETS:
            ds.output_file_names["test"][tgt] = output_paths[tgt]

        ds.write_test_sets()

        for tgt in TARGETS:
            assert os.path.exists(output_paths[tgt])
            os.remove(output_paths[tgt])  # comment out to debug
