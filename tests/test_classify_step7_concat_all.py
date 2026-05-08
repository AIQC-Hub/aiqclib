"""
Unit tests for the ConcatDataSetAll class, which handles concatenating prediction datasets
generated in earlier classification steps.
"""

import os
import unittest
from pathlib import Path

import polars as pl

from aiqclib.classify.step7_concat_datasets.dataset_all import ConcatDataSetAll
from aiqclib.common.config.classify_config import ClassificationConfig
from aiqclib.common.loader.classify_loader import (
    load_classify_step1_input_dataset,
    load_classify_step2_summary_dataset,
    load_classify_step3_select_dataset,
    load_classify_step4_locate_dataset,
    load_classify_step5_extract_dataset,
    load_classify_step6_classify_dataset,
)


class TestConcatPredictions(unittest.TestCase):
    """
    A suite of tests for the ConcatDataSetAll class, ensuring proper concatenation
    and handling of prediction datasets.
    """

    def setUp(self):
        """
        Set up test environment by loading necessary configuration and preparing
        input data, summary statistics, selected profiles, located data, extracted features,
        and classified predictions from previous steps.
        """
        self.config_file_path = str(
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_classify_001.yaml"
        )
        self.config = ClassificationConfig(str(self.config_file_path))
        self.config.select("NRT_BO_001")

        model_path = Path(__file__).resolve().parent / "data" / "training"
        self.model_file_names = {
            "temp": str(model_path / "model_temp.joblib"),
            "psal": str(model_path / "model_psal.joblib"),
            "pres": str(model_path / "model_pres.joblib"),
        }

        self.prediction_file_name = str(
            Path(__file__).resolve().parent
            / "data"
            / "classify"
            / "temp_predictions.parquet"
        )

        self.test_data_file = str(
            Path(__file__).resolve().parent
            / "data"
            / "input"
            / "nrt_cora_bo_test.parquet"
        )
        self.ds_input = load_classify_step1_input_dataset(self.config)
        self.ds_input.input_file_name = str(self.test_data_file)
        self.ds_input.read_input_data()

        self.ds_summary = load_classify_step2_summary_dataset(
            self.config, input_data=self.ds_input.input_data
        )
        self.ds_summary.calculate_stats()

        self.ds_select = load_classify_step3_select_dataset(
            self.config, input_data=self.ds_input.input_data
        )
        self.ds_select.label_profiles()

        self.ds_locate = load_classify_step4_locate_dataset(
            self.config,
            input_data=self.ds_input.input_data,
            selected_profiles=self.ds_select.selected_profiles,
        )
        self.ds_locate.process_targets()

        self.ds_extract = load_classify_step5_extract_dataset(
            self.config,
            input_data=self.ds_input.input_data,
            selected_profiles=self.ds_select.selected_profiles,
            selected_rows=self.ds_locate.selected_rows,
            summary_stats=self.ds_summary.summary_stats,
        )
        self.ds_extract.process_targets()

        self.ds_classify = load_classify_step6_classify_dataset(
            self.config, self.ds_extract.target_features
        )
        self.ds_classify.model_file_names = self.model_file_names
        self.ds_classify.read_models()
        self.ds_classify.test_targets()

    def test_step_name(self):
        """Check that the ConcatDataSetAll step name is correctly assigned."""
        ds = ConcatDataSetAll(self.config)
        self.assertEqual(ds.step_name, "concat")

    def test_output_file_names(self):
        """Verify that the default output file name for merged predictions is as expected from config."""
        ds = ConcatDataSetAll(self.config)

        self.assertEqual(
            "/path/to/concat_1/nrt_bo_001/concat_folder_1/predictions.parquet",
            str(ds.output_file_name),
        )

    def test_test_sets(self):
        """
        Check that input data and predictions are correctly loaded into ConcatDataSetAll
        and have the expected shapes for further processing.
        """
        ds = ConcatDataSetAll(
            self.config,
            input_data=self.ds_input.input_data,
            predictions=self.ds_classify.predictions,
        )

        self.assertIsInstance(ds.input_data, pl.DataFrame)
        self.assertEqual(ds.input_data.shape[0], 19480)
        self.assertEqual(ds.input_data.shape[1], 30)

        self.assertIsInstance(ds.predictions["temp"], pl.DataFrame)
        self.assertEqual(ds.predictions["temp"].shape[0], 19480)
        self.assertEqual(ds.predictions["temp"].shape[1], 7)

        self.assertIsInstance(ds.predictions["psal"], pl.DataFrame)
        self.assertEqual(ds.predictions["psal"].shape[0], 19480)
        self.assertEqual(ds.predictions["psal"].shape[1], 7)

        self.assertIsInstance(ds.predictions["pres"], pl.DataFrame)
        self.assertEqual(ds.predictions["pres"].shape[0], 19480)
        self.assertEqual(ds.predictions["pres"].shape[1], 7)

    def test_merge_predictions(self):
        """
        Confirm that merging predictions correctly combines input data and
        individual parameter predictions into a single Polars DataFrame with the expected shape.
        """
        ds = ConcatDataSetAll(
            self.config,
            input_data=self.ds_input.input_data,
            predictions=self.ds_classify.predictions,
        )
        ds.merge_predictions()

        self.assertIsInstance(ds.merged_predictions, pl.DataFrame)
        self.assertEqual(ds.merged_predictions.shape[0], 19480)
        self.assertEqual(ds.merged_predictions.shape[1], 39)

    def test_merge_predictions_with_empty_input(self):
        """
        Check that a ValueError is raised if the input data for merging is absent (None).
        """
        ds = ConcatDataSetAll(
            self.config,
            input_data=None,
            predictions=self.ds_classify.predictions,
        )
        with self.assertRaises(ValueError):
            ds.merge_predictions()

    def test_merge_predictions_with_empty_predictions(self):
        """
        Check that a ValueError is raised if the predictions for merging are absent (None).
        """
        ds = ConcatDataSetAll(
            self.config,
            input_data=self.ds_input.input_data,
            predictions=None,
        )
        with self.assertRaises(ValueError):
            ds.merge_predictions()

    def test_write_predictions(self):
        """
        Check that the merged predictions are correctly written to a Parquet file
        and that the file exists afterwards, then clean up the created file.
        """
        ds = ConcatDataSetAll(
            self.config,
            input_data=self.ds_input.input_data,
            predictions=self.ds_classify.predictions,
        )

        data_path = Path(__file__).resolve().parent / "data" / "classify"
        ds.output_file_name = str(data_path / "temp_predictions.parquet")

        ds.merge_predictions()
        ds.write_merged_predictions()

        self.assertTrue(os.path.exists(ds.output_file_name))

        os.remove(ds.output_file_name)

    def test_write_no_results(self):
        """
        Ensure ValueError is raised if write_merged_predictions is called
        before predictions have been merged, as `merged_predictions` would be None.
        """
        ds = ConcatDataSetAll(
            self.config,
            input_data=self.ds_input.input_data,
            predictions=self.ds_classify.predictions,
        )

        with self.assertRaises(ValueError):
            ds.write_merged_predictions()
