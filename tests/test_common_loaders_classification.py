"""
This module contains unit tests for the classification dataset loader functions.

It verifies that each `load_classify_stepX_dataset` function correctly
instantiates the appropriate dataset class and properly handles the
injection of input data and intermediate results from previous steps.
"""

import unittest
from pathlib import Path

import polars as pl

from aiqclib.classify.step1_read_input.dataset_all import InputDataSetAll
from aiqclib.classify.step2_calc_stats.dataset_all import SummaryDataSetAll
from aiqclib.classify.step3_select_profiles.dataset_all import SelectDataSetAll
from aiqclib.classify.step4_select_rows.dataset_all import LocateDataSetAll
from aiqclib.classify.step5_extract_features.dataset_all import ExtractDataSetAll
from aiqclib.classify.step6_classify_dataset.dataset_all import ClassifyAll
from aiqclib.classify.step6_classify_dataset.dataset_all_suite import ClassifyAllSuite
from aiqclib.classify.step7_concat_datasets.dataset_all import ConcatDataSetAll
from aiqclib.classify.step7_concat_datasets.dataset_suite import ConcatDataSetSuite
from aiqclib.common.config.classify_config import ClassificationConfig
from aiqclib.common.loader.classify_loader import (
    load_classify_step1_input_dataset,
    load_classify_step2_summary_dataset,
    load_classify_step3_select_dataset,
    load_classify_step4_locate_dataset,
    load_classify_step5_extract_dataset,
    load_classify_step6_classify_dataset,
    load_classify_step7_concat_dataset,
)


class TestClassifyInputClassLoader(unittest.TestCase):
    """
    Tests related to loading the InputDataSetAll class.
    """

    def setUp(self):
        """
        Define the path to the test config file and select a dataset
        prior to each test.
        """
        self.config_file_path = str(
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_classify_001.yaml"
        )
        self.config = ClassificationConfig(str(self.config_file_path))
        self.config.select("NRT_BO_001")

    def test_load_dataset_valid_config(self):
        """
        Check that load_classify_step1_input_dataset returns an InputDataSetAll instance
        with the expected step name.
        """
        ds = load_classify_step1_input_dataset(self.config)
        self.assertIsInstance(ds, InputDataSetAll)
        self.assertEqual(ds.step_name, "input")

    def test_load_input_class_with_invalid_config(self):
        """
        Ensure that an invalid input class name configured in the YAML raises a ValueError.
        """
        self.config.data["step_class_set"]["steps"]["input"] = "InvalidClass"
        with self.assertRaises(ValueError):
            _ = load_classify_step1_input_dataset(self.config)


class TestClassifySummaryClassLoader(unittest.TestCase):
    """
    Tests related to loading the SummaryDataSetAll class.
    """

    def setUp(self):
        """
        Define the path to the test config file and select a dataset
        prior to each test.
        """
        self.config_file_path = str(
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_classify_001.yaml"
        )
        self.config = ClassificationConfig(str(self.config_file_path))
        self.config.select("NRT_BO_001")
        self.test_data_file = (
            Path(__file__).resolve().parent
            / "data"
            / "input"
            / "nrt_cora_bo_test.parquet"
        )

    def test_load_dataset_valid_config(self):
        """
        Check that load_classify_step2_summary_dataset returns a SummaryDataSetAll instance
        with the correct step name.
        """
        ds = load_classify_step2_summary_dataset(self.config)
        self.assertIsInstance(ds, SummaryDataSetAll)
        self.assertEqual(ds.step_name, "summary")

    def test_load_dataset_input_data(self):
        """
        Check that load_classify_step2_summary_dataset properly receives and sets the
        'input_data' attribute when provided during loading.
        """
        ds_input = load_classify_step1_input_dataset(self.config)
        ds_input.input_file_name = str(self.test_data_file)
        ds_input.read_input_data()

        ds = load_classify_step2_summary_dataset(self.config, ds_input.input_data)
        self.assertIsInstance(ds, SummaryDataSetAll)
        self.assertIsInstance(ds.input_data, pl.DataFrame)
        self.assertEqual(ds.input_data.shape[0], 19480)
        self.assertEqual(ds.input_data.shape[1], 30)


class TestClassifySelectClassLoader(unittest.TestCase):
    """
    Tests related to loading the SelectDataSetAll class.
    """

    def setUp(self):
        """
        Define the path to the test config file and select a dataset
        prior to each test.
        """
        self.config_file_path = str(
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_classify_001.yaml"
        )
        self.config = ClassificationConfig(str(self.config_file_path))
        self.config.select("NRT_BO_001")
        self.test_data_file = (
            Path(__file__).resolve().parent
            / "data"
            / "input"
            / "nrt_cora_bo_test.parquet"
        )

    def test_load_dataset_valid_config(self):
        """
        Check that load_classify_step3_select_dataset returns a SelectDataSetAll instance
        with the correct step name.
        """
        ds = load_classify_step3_select_dataset(self.config)
        self.assertIsInstance(ds, SelectDataSetAll)
        self.assertEqual(ds.step_name, "select")

    def test_load_dataset_input_data(self):
        """
        Check that load_classify_step3_select_dataset properly receives and sets the
        'input_data' attribute when provided during loading.
        """
        ds_input = load_classify_step1_input_dataset(self.config)
        ds_input.input_file_name = str(self.test_data_file)
        ds_input.read_input_data()

        ds = load_classify_step3_select_dataset(self.config, ds_input.input_data)
        self.assertIsInstance(ds, SelectDataSetAll)
        self.assertIsInstance(ds.input_data, pl.DataFrame)
        self.assertEqual(ds.input_data.shape[0], 19480)
        self.assertEqual(ds.input_data.shape[1], 30)


class TestClassifyLocateClassLoader(unittest.TestCase):
    """
    Tests related to loading the LocateDataSetAll class.
    """

    def setUp(self):
        """
        Define the path to the test config file and select a dataset
        prior to each test.
        """
        self.config_file_path = str(
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_classify_001.yaml"
        )
        self.config = ClassificationConfig(str(self.config_file_path))
        self.config.select("NRT_BO_001")
        self.test_data_file = (
            Path(__file__).resolve().parent
            / "data"
            / "input"
            / "nrt_cora_bo_test.parquet"
        )

    def test_load_dataset_valid_config(self):
        """
        Check that load_classify_step4_locate_dataset returns a LocateDataSetAll instance
        with the correct step name.
        """
        ds = load_classify_step4_locate_dataset(self.config)
        self.assertIsInstance(ds, LocateDataSetAll)
        self.assertEqual(ds.step_name, "locate")

    def test_load_dataset_input_data_and_profiles(self):
        """
        Check that load_classify_step4_locate_dataset properly receives and sets the
        'input_data' and 'selected_profiles' attributes when provided.
        """
        ds_input = load_classify_step1_input_dataset(self.config)
        ds_input.input_file_name = str(self.test_data_file)
        ds_input.read_input_data()

        ds_select = load_classify_step3_select_dataset(self.config, ds_input.input_data)
        ds_select.label_profiles()

        ds = load_classify_step4_locate_dataset(
            self.config, ds_input.input_data, ds_select.selected_profiles
        )

        self.assertIsInstance(ds, LocateDataSetAll)

        self.assertIsInstance(ds.input_data, pl.DataFrame)
        self.assertEqual(ds.input_data.shape[0], 19480)
        self.assertEqual(ds.input_data.shape[1], 30)

        self.assertIsInstance(ds.selected_profiles, pl.DataFrame)
        self.assertEqual(ds.selected_profiles.shape[0], 84)
        self.assertEqual(ds.selected_profiles.shape[1], 8)


class TestClassifyExtractClassLoader(unittest.TestCase):
    """
    Tests related to loading the ExtractDataSetAll class.
    """

    def setUp(self):
        """
        Define the path to the test config file and select a dataset
        prior to each test.
        """
        self.config_file_path = str(
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_classify_001.yaml"
        )
        self.config = ClassificationConfig(str(self.config_file_path))
        self.config.select("NRT_BO_001")
        self.test_data_file = (
            Path(__file__).resolve().parent
            / "data"
            / "input"
            / "nrt_cora_bo_test.parquet"
        )

    def test_load_dataset_valid_config(self):
        """
        Check that load_classify_step5_extract_dataset returns an ExtractDataSetAll instance
        with the correct step name.
        """
        ds = load_classify_step5_extract_dataset(self.config)
        self.assertIsInstance(ds, ExtractDataSetAll)
        self.assertEqual(ds.step_name, "extract")

    def test_load_dataset_input_data_and_profiles(self):
        """
        Check that load_classify_step5_extract_dataset properly receives and sets
        'input_data', 'selected_profiles', 'selected_rows', and 'summary_stats'
        attributes when provided. Also verifies 'filtered_input' from internal processing.
        """
        ds_input = load_classify_step1_input_dataset(self.config)
        ds_input.input_file_name = str(self.test_data_file)
        ds_input.read_input_data()

        ds_select = load_classify_step3_select_dataset(self.config, ds_input.input_data)
        ds_select.label_profiles()

        ds_summary = load_classify_step2_summary_dataset(
            self.config, ds_input.input_data
        )
        ds_summary.calculate_stats()

        ds_locate = load_classify_step4_locate_dataset(
            self.config, ds_input.input_data, ds_select.selected_profiles
        )
        ds_locate.process_targets()

        ds = load_classify_step5_extract_dataset(
            self.config,
            ds_input.input_data,
            ds_select.selected_profiles,
            ds_locate.selected_rows,
            ds_summary.summary_stats,
        )

        self.assertIsInstance(ds, ExtractDataSetAll)

        self.assertIsInstance(ds.input_data, pl.DataFrame)
        self.assertEqual(ds.input_data.shape[0], 19480)
        self.assertEqual(ds.input_data.shape[1], 30)

        self.assertIsInstance(ds.summary_stats, pl.DataFrame)
        self.assertEqual(ds.summary_stats.shape[0], 425)
        self.assertEqual(ds.summary_stats.shape[1], 12)

        self.assertIsInstance(ds.selected_profiles, pl.DataFrame)
        self.assertEqual(ds.selected_profiles.shape[0], 84)
        self.assertEqual(ds.selected_profiles.shape[1], 8)

        self.assertIsInstance(ds.filtered_input, pl.DataFrame)
        self.assertEqual(ds.filtered_input.shape[0], 19480)
        self.assertEqual(ds.filtered_input.shape[1], 30)

        self.assertIsInstance(ds.selected_rows["temp"], pl.DataFrame)
        self.assertEqual(ds.selected_rows["temp"].shape[0], 19480)
        self.assertEqual(ds.selected_rows["temp"].shape[1], 9)

        self.assertIsInstance(ds.selected_rows["psal"], pl.DataFrame)
        self.assertEqual(ds.selected_rows["psal"].shape[0], 19480)
        self.assertEqual(ds.selected_rows["psal"].shape[1], 9)


class TestClassifyClassifyClassLoader(unittest.TestCase):
    """
    Tests related to loading the ClassifyAll class.
    """

    def setUp(self):
        """
        Define the path to the test config file and select a dataset
        prior to each test.
        """
        self.config_file_path = str(
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_classify_001.yaml"
        )
        self.config = ClassificationConfig(str(self.config_file_path))
        self.config.select("NRT_BO_001")
        self.test_data_file = (
            Path(__file__).resolve().parent
            / "data"
            / "input"
            / "nrt_cora_bo_test.parquet"
        )

    def test_load_dataset_valid_config(self):
        """
        Check that load_classify_step6_classify_dataset returns a ClassifyAll instance
        with the correct step name.
        """
        ds = load_classify_step6_classify_dataset(self.config)
        self.assertIsInstance(ds, ClassifyAll)
        self.assertEqual(ds.step_name, "classify")

    def test_load_classifiction_suite_dataset(self):
        """
        Check that load_classify_step6_classify_dataset returns a ClassifyAllSuite instance
        with the correct step name.
        """
        self.config.data["step_class_set"]["steps"]["model"] = "ModelSuite"
        self.config.data["step_class_set"]["steps"]["classify"] = "ClassifyAllSuite"
        ds = load_classify_step6_classify_dataset(self.config)
        self.assertIsInstance(ds, ClassifyAllSuite)
        self.assertEqual(ds.step_name, "classify")

    def test_load_dataset_input_data(self):
        """
        Check that load_classify_step6_classify_dataset properly receives and sets
        the 'target_features' attribute (which populates 'test_sets' internally)
        when provided, after all prior steps have generated the necessary data.
        """
        ds_input = load_classify_step1_input_dataset(self.config)
        ds_input.input_file_name = str(self.test_data_file)
        ds_input.read_input_data()

        ds_select = load_classify_step3_select_dataset(self.config, ds_input.input_data)
        ds_select.label_profiles()

        ds_summary = load_classify_step2_summary_dataset(
            self.config, ds_input.input_data
        )
        ds_summary.calculate_stats()

        ds_locate = load_classify_step4_locate_dataset(
            self.config, ds_input.input_data, ds_select.selected_profiles
        )
        ds_locate.process_targets()

        ds_extract = load_classify_step5_extract_dataset(
            self.config,
            ds_input.input_data,
            ds_select.selected_profiles,
            ds_locate.selected_rows,
            ds_summary.summary_stats,
        )
        ds_extract.process_targets()

        ds = load_classify_step6_classify_dataset(
            self.config, ds_extract.target_features
        )

        self.assertIsInstance(ds, ClassifyAll)

        self.assertIsInstance(ds.test_sets["temp"], pl.DataFrame)
        self.assertEqual(ds.test_sets["temp"].shape[0], 19480)
        self.assertEqual(ds.test_sets["temp"].shape[1], 56)

        self.assertIsInstance(ds.test_sets["psal"], pl.DataFrame)
        self.assertEqual(ds.test_sets["psal"].shape[0], 19480)
        self.assertEqual(ds.test_sets["psal"].shape[1], 56)

    def test_load_suite_dataset_input_data(self):
        """
        Check that load_classify_step6_classify_dataset properly receives and sets
        the 'target_features' attribute (which populates 'test_sets' internally)
        when provided, after all prior steps have generated the necessary data.
        """
        self.config.data["step_class_set"]["steps"]["model"] = "ModelSuite"
        self.config.data["step_class_set"]["steps"]["classify"] = "ClassifyAllSuite"

        ds_input = load_classify_step1_input_dataset(self.config)
        ds_input.input_file_name = str(self.test_data_file)
        ds_input.read_input_data()

        ds_select = load_classify_step3_select_dataset(self.config, ds_input.input_data)
        ds_select.label_profiles()

        ds_summary = load_classify_step2_summary_dataset(
            self.config, ds_input.input_data
        )
        ds_summary.calculate_stats()

        ds_locate = load_classify_step4_locate_dataset(
            self.config, ds_input.input_data, ds_select.selected_profiles
        )
        ds_locate.process_targets()

        ds_extract = load_classify_step5_extract_dataset(
            self.config,
            ds_input.input_data,
            ds_select.selected_profiles,
            ds_locate.selected_rows,
            ds_summary.summary_stats,
        )
        ds_extract.process_targets()

        ds = load_classify_step6_classify_dataset(
            self.config, ds_extract.target_features
        )

        self.assertIsInstance(ds, ClassifyAllSuite)

        self.assertIsInstance(ds.test_sets["temp"], pl.DataFrame)
        self.assertEqual(ds.test_sets["temp"].shape[0], 19480)
        self.assertEqual(ds.test_sets["temp"].shape[1], 56)

        self.assertIsInstance(ds.test_sets["psal"], pl.DataFrame)
        self.assertEqual(ds.test_sets["psal"].shape[0], 19480)
        self.assertEqual(ds.test_sets["psal"].shape[1], 56)


class TestClassifyConcatClassLoader(unittest.TestCase):
    """
    Tests related to loading the ConcatDataSetAll class.
    """

    def setUp(self):
        """
        Define the path to the test config file and select a dataset
        prior to each test.
        """
        self.config_file_path = str(
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_classify_001.yaml"
        )
        self.config = ClassificationConfig(str(self.config_file_path))
        self.config.select("NRT_BO_001")
        self.test_data_file = (
            Path(__file__).resolve().parent
            / "data"
            / "input"
            / "nrt_cora_bo_test.parquet"
        )

        model_path = Path(__file__).resolve().parent / "data" / "training"
        self.model_file_names = {
            "temp": str(model_path / "model_temp.joblib"),
            "psal": str(model_path / "model_psal.joblib"),
            "pres": str(model_path / "model_pres.joblib"),
        }
        self.suite_model_file_names = {
            "xgb_temp": str(model_path / "model_temp_xgb.joblib"),
            "dt_temp": str(model_path / "model_temp_dt.joblib"),
            "xgb_psal": str(model_path / "model_psal_xgb.joblib"),
            "dt_psal": str(model_path / "model_psal_dt.joblib"),
            "xgb_pres": str(model_path / "model_pres_xgb.joblib"),
            "dt_pres": str(model_path / "model_pres_dt.joblib"),
        }

    def test_load_dataset_valid_config(self):
        """
        Check that load_classify_step7_concat_dataset returns a ConcatDataSetAll instance
        with the correct step name.
        """
        ds = load_classify_step7_concat_dataset(self.config)
        self.assertIsInstance(ds, ConcatDataSetAll)
        self.assertEqual(ds.step_name, "concat")

    def test_load_suite_dataset_valid_config(self):
        """
        Check that load_classify_step7_concat_dataset returns a ConcatDataSetSuite instance
        with the correct step name.
        """
        self.config.data["step_class_set"]["steps"]["model"] = "ModelSuite"
        self.config.data["step_class_set"]["steps"]["classify"] = "ClassifyAllSuite"
        self.config.data["step_class_set"]["steps"]["concat"] = "ConcatDataSetSuite"
        ds = load_classify_step7_concat_dataset(self.config)
        self.assertIsInstance(ds, ConcatDataSetSuite)
        self.assertEqual(ds.step_name, "concat")

    def test_load_dataset_input_data(self):
        """
        Check that load_classify_step7_concat_dataset properly receives and sets
        the 'input_data' and 'predictions' attributes when provided, after
        all prior steps have generated the necessary data.
        """
        ds_input = load_classify_step1_input_dataset(self.config)
        ds_input.input_file_name = str(self.test_data_file)
        ds_input.read_input_data()

        ds_select = load_classify_step3_select_dataset(self.config, ds_input.input_data)
        ds_select.label_profiles()

        ds_summary = load_classify_step2_summary_dataset(
            self.config, ds_input.input_data
        )
        ds_summary.calculate_stats()

        ds_locate = load_classify_step4_locate_dataset(
            self.config, ds_input.input_data, ds_select.selected_profiles
        )
        ds_locate.process_targets()

        ds_extract = load_classify_step5_extract_dataset(
            self.config,
            ds_input.input_data,
            ds_select.selected_profiles,
            ds_locate.selected_rows,
            ds_summary.summary_stats,
        )
        ds_extract.process_targets()

        ds_classify = load_classify_step6_classify_dataset(
            self.config, ds_extract.target_features
        )
        ds_classify.model_file_names = self.model_file_names
        ds_classify.read_models()
        ds_classify.test_targets()

        ds = load_classify_step7_concat_dataset(
            self.config, ds_input.input_data, ds_classify.predictions
        )

        self.assertIsInstance(ds, ConcatDataSetAll)

        self.assertIsInstance(ds.predictions["temp"], pl.DataFrame)
        self.assertEqual(ds.predictions["temp"].shape[0], 19480)
        self.assertEqual(ds.predictions["temp"].shape[1], 7)

        self.assertIsInstance(ds.predictions["psal"], pl.DataFrame)
        self.assertEqual(ds.predictions["psal"].shape[0], 19480)
        self.assertEqual(ds.predictions["psal"].shape[1], 7)

    def test_load_suite_dataset_input_data(self):
        """
        Check that load_classify_step7_concat_dataset properly receives and sets
        the 'input_data' and 'predictions' attributes when provided, after
        all prior steps have generated the necessary data.
        """
        self.config.data["step_class_set"]["steps"]["model"] = "ModelSuite"
        self.config.data["step_class_set"]["steps"]["classify"] = "ClassifyAllSuite"
        self.config.data["step_class_set"]["steps"]["concat"] = "ConcatDataSetSuite"
        self.config.data["step_class_set"]["steps"]["model"] = "ModelSuite"
        self.config.data["step_param_set"]["steps"]["model"] = {
            "methods": ["XGB", "DT"]
        }

        ds_input = load_classify_step1_input_dataset(self.config)
        ds_input.input_file_name = str(self.test_data_file)
        ds_input.read_input_data()

        ds_select = load_classify_step3_select_dataset(self.config, ds_input.input_data)
        ds_select.label_profiles()

        ds_summary = load_classify_step2_summary_dataset(
            self.config, ds_input.input_data
        )
        ds_summary.calculate_stats()

        ds_locate = load_classify_step4_locate_dataset(
            self.config, ds_input.input_data, ds_select.selected_profiles
        )
        ds_locate.process_targets()

        ds_extract = load_classify_step5_extract_dataset(
            self.config,
            ds_input.input_data,
            ds_select.selected_profiles,
            ds_locate.selected_rows,
            ds_summary.summary_stats,
        )
        ds_extract.process_targets()

        ds_classify = load_classify_step6_classify_dataset(
            self.config, ds_extract.target_features
        )
        ds_classify.model_file_names = self.suite_model_file_names
        ds_classify.read_models()
        ds_classify.test_targets()

        ds = load_classify_step7_concat_dataset(
            self.config, ds_input.input_data, ds_classify.predictions
        )

        self.assertIsInstance(ds, ConcatDataSetSuite)

        self.assertIsInstance(ds.predictions["temp"], pl.DataFrame)
        self.assertEqual(ds.predictions["temp"].shape[0], 19480 * 2)
        self.assertEqual(ds.predictions["temp"].shape[1], 8)

        self.assertIsInstance(ds.predictions["psal"], pl.DataFrame)
        self.assertEqual(ds.predictions["psal"].shape[0], 19480 * 2)
        self.assertEqual(ds.predictions["psal"].shape[1], 8)
