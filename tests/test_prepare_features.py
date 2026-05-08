"""
This module contains unit tests for various feature extraction classes
within the aiqclib library. It uses a shared base class to set up
common data loading and validation steps, ensuring that individual
feature implementations correctly process and prepare data.
"""

import unittest
from pathlib import Path

import polars as pl

from aiqclib.common.config.dataset_config import DataSetConfig
from aiqclib.common.loader.dataset_loader import (
    load_step1_input_dataset,
    load_step2_summary_dataset,
    load_step3_select_dataset,
    load_step4_locate_dataset,
    load_step5_extract_dataset,
)
from aiqclib.prepare.features.basic_values import BasicValues
from aiqclib.prepare.features.day_of_year import DayOfYearFeat
from aiqclib.prepare.features.location import LocationFeat
from aiqclib.prepare.features.profile_summary import ProfileSummaryStats


class _TestFeatureBase(unittest.TestCase):
    """
    A base class providing shared setup and validation logic for feature-based tests.
    """

    def _setup(self, class_name):
        """
        Loads necessary data from multiple processing steps (input, summary, select, locate, extract)
        to prepare the environment for feature extraction tests. This method initializes
        `self.config`, `self.ds_input`, `self.ds_summary`, `self.ds_select`,
        `self.ds_locate`, and `self.ds_extract` attributes.
        """
        self.config_file_path = (
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_dataset_001.yaml"
        )
        self.config = DataSetConfig(str(self.config_file_path))
        self.config.select("NRT_BO_001")
        self.test_data_file = (
            Path(__file__).resolve().parent
            / "data"
            / "input"
            / "nrt_cora_bo_test.parquet"
        )
        self.ds_input = load_step1_input_dataset(self.config)
        self.ds_input.input_file_name = str(self.test_data_file)
        self.ds_input.read_input_data()

        self.ds_summary = load_step2_summary_dataset(
            self.config, self.ds_input.input_data
        )
        self.ds_summary.calculate_stats()

        self.ds_select = load_step3_select_dataset(
            self.config, self.ds_input.input_data
        )
        self.ds_select.label_profiles()

        self.ds_locate = load_step4_locate_dataset(
            self.config, self.ds_input.input_data, self.ds_select.selected_profiles
        )
        self.ds_locate.process_targets()

        self.ds_extract = load_step5_extract_dataset(
            self.config,
            self.ds_input.input_data,
            self.ds_select.selected_profiles,
            self.ds_locate.selected_rows,
            self.ds_summary.summary_stats,
        )

        self.class_name = class_name

    def _test_init_arguments(self, feature_info):
        """
        Validates that required input dataframes (selected_profiles, filtered_input,
        selected_rows, summary_stats) are correctly initialized as Polars DataFrames
        and have the expected dimensions within the feature class constructor.
        """
        ds = self.class_name(
            "temp",
            feature_info,
            self.ds_select.selected_profiles,
            self.ds_extract.filtered_input,
            self.ds_locate.selected_rows,
            self.ds_summary.summary_stats,
        )

        self.assertIsInstance(ds.selected_profiles, pl.DataFrame)
        self.assertEqual(ds.selected_profiles.shape[0], 50)
        self.assertEqual(ds.selected_profiles.shape[1], 8)

        self.assertIsInstance(ds.filtered_input, pl.DataFrame)
        self.assertEqual(ds.filtered_input.shape[0], 10683)
        self.assertEqual(ds.filtered_input.shape[1], 30)

        self.assertIsInstance(ds.selected_rows["temp"], pl.DataFrame)
        self.assertEqual(ds.selected_rows["temp"].shape[0], 128)
        self.assertEqual(ds.selected_rows["temp"].shape[1], 9)

        self.assertIsInstance(ds.selected_rows["psal"], pl.DataFrame)
        self.assertEqual(ds.selected_rows["psal"].shape[0], 140)
        self.assertEqual(ds.selected_rows["psal"].shape[1], 9)

        self.assertIsInstance(ds.summary_stats, pl.DataFrame)
        self.assertEqual(ds.summary_stats.shape[0], 2520)
        self.assertEqual(ds.summary_stats.shape[1], 12)


class TestLocationFeature(_TestFeatureBase):
    """
    Tests for verifying the LocationFeat class, ensuring location-based features
    are extracted and scaled as expected.
    """

    def setUp(self):
        """
        Initializes the test environment for LocationFeat, loading necessary data
        and configuring the specific feature information.
        """
        super()._setup(LocationFeat)
        self.feature_info = {
            "class": "location",
            "stats": {
                "longitude": {"min": 14.5, "max": 23.5},
                "latitude": {"min": 55, "max": 66},
            },
            "col_names": ["longitude", "latitude"],
            "stats_set": {"type": "min_max", "name": "location"},
        }

    def test_init_arguments(self):
        """
        Checks the initialization of required data for the LocationFeat class
        by calling the shared validation method from the base class.
        """
        super()._test_init_arguments(self.feature_info)

    def test_location_features(self):
        """
        Verifies that location-based features (longitude, latitude) are correctly
        extracted and scaled, checking the resulting DataFrame type and dimensions.
        """
        ds = LocationFeat(
            "temp",
            self.feature_info,
            self.ds_select.selected_profiles,
            self.ds_extract.filtered_input,
            self.ds_locate.selected_rows,
            self.ds_summary.summary_stats,
        )
        ds.extract_features()
        ds.scale_second()

        self.assertIsInstance(ds.features, pl.DataFrame)
        self.assertEqual(ds.features.shape[0], 128)
        self.assertEqual(ds.features.shape[1], 3)


class TestDayOfYearFeature(_TestFeatureBase):
    """
    Tests for verifying the DayOfYearFeat class, ensuring date-based features
    (with optional sine conversion) are extracted and scaled as expected.
    """

    def setUp(self):
        """
        Initializes the test environment for DayOfYearFeat, loading necessary data
        and configuring the specific feature information, including sine conversion.
        """
        super()._setup(DayOfYearFeat)
        self.feature_info = {
            "class": "day_of_year",
            "convert": "sine",
        }

    def test_init_arguments(self):
        """
        Checks the initialization of required data for the DayOfYearFeat class
        by calling the shared validation method from the base class.
        """
        super()._test_init_arguments(self.feature_info)

    def test_day_of_year_features(self):
        """
        Verifies that day-of-year features are correctly extracted and scaled,
        specifically testing with sine conversion, and checking the resulting
        DataFrame type and dimensions.
        """
        ds = DayOfYearFeat(
            "temp",
            self.feature_info,
            self.ds_select.selected_profiles,
            self.ds_extract.filtered_input,
            self.ds_locate.selected_rows,
            self.ds_summary.summary_stats,
        )
        ds.extract_features()
        ds.scale_second()

        self.assertIsInstance(ds.features, pl.DataFrame)
        self.assertEqual(ds.features.shape[0], 128)
        self.assertEqual(ds.features.shape[1], 2)

    def test_day_of_year_features_no_param(self):
        """
        Verifies that the `scale_second` method does not modify features
        when `feature_info` is set to `None`, implying no scaling parameters.
        """
        ds = DayOfYearFeat(
            "temp",
            self.feature_info,
            self.ds_select.selected_profiles,
            self.ds_extract.filtered_input,
            self.ds_locate.selected_rows,
            self.ds_summary.summary_stats,
        )
        ds.extract_features()
        features = ds.features
        ds.feature_info = None  # Simulate missing feature info for scaling
        ds.scale_second()
        self.assertTrue(ds.features.equals(features))

    def test_day_of_year_features_no_convert_param(self):
        """
        Verifies that the `scale_second` method does not modify features
        when the 'convert' parameter is missing from `feature_info`,
        implying no specific conversion/scaling is applied.
        """
        ds = DayOfYearFeat(
            "temp",
            self.feature_info,
            self.ds_select.selected_profiles,
            self.ds_extract.filtered_input,
            self.ds_locate.selected_rows,
            self.ds_summary.summary_stats,
        )
        ds.extract_features()
        features = ds.features
        ds.feature_info = {
            "class": "day_of_year",
        }  # Simulate missing 'convert' parameter
        ds.scale_second()
        self.assertTrue(ds.features.equals(features))

    def test_convert_cosine(self):
        """
        Verifies that day-of-year features are correctly extracted and scaled,
        specifically testing with sine conversion, and checking the resulting
        DataFrame type and dimensions.
        """
        ds = DayOfYearFeat(
            "temp",
            self.feature_info,
            self.ds_select.selected_profiles,
            self.ds_extract.filtered_input,
            self.ds_locate.selected_rows,
            self.ds_summary.summary_stats,
        )
        ds.feature_info = {
            "class": "day_of_year",
            "convert": "cosine",
        }
        ds.extract_features()
        ds.scale_second()

        self.assertIsInstance(ds.features, pl.DataFrame)
        self.assertEqual(ds.features.shape[0], 128)
        self.assertEqual(ds.features.shape[1], 2)


class TestProfileSummaryStatsFeature(_TestFeatureBase):
    """
    Tests for verifying the ProfileSummaryStats class, which computes advanced
    summary statistics for multiple variables (temp, psal, pres).
    """

    def setUp(self):
        """
        Initializes the test environment for ProfileSummaryStats, loading necessary
        data and configuring the specific feature information with detailed stats
        for multiple variables.
        """
        super()._setup(ProfileSummaryStats)
        self.feature_info = {
            "class": "profile_summary_stats",
            "stats": {
                "temp": {
                    "mean": {"min": 0, "max": 12.5},
                    "median": {"min": 0, "max": 15},
                    "sd": {"min": 0, "max": 6.5},
                    "pct25": {"min": 0, "max": 12},
                    "pct75": {"min": 1, "max": 19},
                },
                "psal": {
                    "mean": {"min": 2.9, "max": 12},
                    "median": {"min": 2.9, "max": 12},
                    "sd": {"min": 0, "max": 4},
                    "pct25": {"min": 2.5, "max": 8.5},
                    "pct75": {"min": 3, "max": 16},
                },
                "pres": {
                    "mean": {"min": 24, "max": 105},
                    "median": {"min": 24, "max": 105},
                    "sd": {"min": 13, "max": 60},
                    "pct25": {"min": 12, "max": 53},
                    "pct75": {"min": 35, "max": 156},
                },
            },
            "col_names": ["temp", "psal", "pres"],
            "stats_set": {"type": "min_max", "name": "profile_summary_stats"},
            "summary_stats_names": ["mean", "median", "sd", "pct25", "pct75"],
        }

    def test_init_arguments(self):
        """
        Checks the initialization of required data for the ProfileSummaryStats class
        by calling the shared validation method from the base class.
        """
        super()._test_init_arguments(self.feature_info)

    def test_profile_summary_stats_features(self):
        """
        Verifies the correct extraction and scaling of extended summary statistics
        (mean, median, standard deviation, 25th percentile, 75th percentile)
        for multiple variables, checking the resulting DataFrame type and dimensions.
        """
        ds = ProfileSummaryStats(
            "temp",
            self.feature_info,
            self.ds_select.selected_profiles,
            self.ds_extract.filtered_input,
            self.ds_locate.selected_rows,
            self.ds_summary.summary_stats,
        )
        ds.extract_features()
        ds.scale_second()

        self.assertIsInstance(ds.features, pl.DataFrame)
        self.assertEqual(ds.features.shape[0], 128)
        self.assertEqual(ds.features.shape[1], 16)


class TestBasicValues3PlusFlanksFeature(_TestFeatureBase):
    """
    Tests for verifying the BasicValues class, specifically its implementation
    for extracting basic statistics and 'flank' values from data points.
    """

    def setUp(self):
        """
        Initializes the test environment for BasicValues, loading
        necessary data and configuring the specific feature information,
        including flank parameters and statistics for variables.
        """
        super()._setup(BasicValues)
        self.feature_info = {
            "class": "basic_values3_plus_flanks",
            "flank_up": 5,
            "stats": {
                "temp": {"min": 0, "max": 20},
                "psal": {"min": 0, "max": 20},
                "pres": {"min": 0, "max": 200},
            },
            "col_names": ["temp", "psal", "pres"],
            "stats_set": {"type": "min_max", "name": "basic_values3_plus_flanks"},
        }

    def test_init_arguments(self):
        """
        Checks the initialization of required data for the BasicValues class
        by calling the shared validation method from the base class.
        """
        super()._test_init_arguments(self.feature_info)

    def test_basic_values3_features(self):
        """
        Validates the extraction and scaling of basic statistics combined with
        'flank' data (values from adjacent points), checking the resulting
        DataFrame type and dimensions.
        """
        ds = BasicValues(
            "temp",
            self.feature_info,
            self.ds_select.selected_profiles,
            self.ds_extract.filtered_input,
            self.ds_locate.selected_rows,
            self.ds_summary.summary_stats,
        )

        ds.scale_first()
        ds.extract_features()

        self.assertIsInstance(ds.features, pl.DataFrame)
        self.assertEqual(ds.features.shape[0], 128)
        self.assertEqual(ds.features.shape[1], 4)
