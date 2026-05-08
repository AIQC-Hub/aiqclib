"""Unit tests for the summary statistics utility functions.

This module contains unit tests for the `get_summary_stats` and
`format_summary_stats` functions, ensuring they correctly calculate and
format statistical summaries from Polars DataFrames.
"""

import unittest
from pathlib import Path

import polars as pl

from aiqclib.interface.stats import get_summary_stats, format_summary_stats


class TestSummaryStats(unittest.TestCase):
    """A test suite for the summary statistics functions.

    This suite includes tests for both `get_summary_stats` and
    `format_summary_stats` functions, covering different types of
    summary statistics (profile-level and global) and various
    formatting options.
    """

    def setUp(self):
        """Set up the test environment by defining the sample input file path.

        This method prepares the path to a sample Parquet data file used
        for testing the summary statistics functions.
        """
        self.test_data_file = (
            Path(__file__).resolve().parent
            / "data"
            / "input"
            / "nrt_cora_bo_test.parquet"
        )

    def test_get_profile_summary_stats(self):
        """Verify that `get_summary_stats` returns correct profile-level statistics.

        This test calls `get_summary_stats` with the "profiles" type and
        asserts that the returned object is a Polars DataFrame.
        """
        ds = get_summary_stats(self.test_data_file, "profiles")
        self.assertIsInstance(ds, pl.DataFrame)

    def test_get_global_summary_stats(self):
        """Verify that `get_summary_stats` returns correct global statistics.

        This test calls `get_summary_stats` with the "all" type and asserts
        that the returned object is a Polars DataFrame.
        """
        ds = get_summary_stats(self.test_data_file, "all")
        self.assertIsInstance(ds, pl.DataFrame)

    def test_format_profile_summary_stats(self):
        """Verify that `format_summary_stats` correctly formats profile-level statistics.

        This test checks the output string for expected variable names and
        statistic types, including filtering by `variables` and `summary_stats`.
        """
        ds = get_summary_stats(self.test_data_file, "profiles")

        stats_str = format_summary_stats(ds)
        self.assertIsInstance(stats_str, str)
        self.assertIn("psal", stats_str)
        self.assertIn("pct25", stats_str)

        stats_str_filtered_vars = format_summary_stats(ds, ["pres", "temp"])
        self.assertIsInstance(stats_str_filtered_vars, str)
        self.assertNotIn("psal", stats_str_filtered_vars)
        self.assertIn("pct25", stats_str_filtered_vars)

        stats_str_filtered_stats = format_summary_stats(ds, ["pres", "temp"], ["mean"])
        self.assertIsInstance(stats_str_filtered_stats, str)
        self.assertNotIn("psal", stats_str_filtered_stats)
        self.assertNotIn("pct25", stats_str_filtered_stats)
        self.assertIn("mean", stats_str_filtered_stats)

    def test_format_global_summary_stats(self):
        """Verify that `format_summary_stats` correctly formats global statistics.

        This test checks the output string for expected variable names,
        including filtering by `variables`.
        """
        ds = get_summary_stats(self.test_data_file, "all")

        stats_str = format_summary_stats(ds)
        self.assertIsInstance(stats_str, str)
        self.assertIn("psal", stats_str)

        stats_str_filtered_vars = format_summary_stats(ds, ["pres", "temp"])
        self.assertIsInstance(stats_str_filtered_vars, str)
        self.assertNotIn("psal", stats_str_filtered_vars)
