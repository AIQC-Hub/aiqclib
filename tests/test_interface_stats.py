"""Unit tests for the ``get_summary_stats`` and ``format_summary_stats`` interface functions.

This file exercises the public-facing summary-stats utility, not the
pipeline-internal ``SummaryDataSetA`` class (that's
``test_prepare_step2_summary_a.py``). The interface functions take a file
path and a "kind" string (``"profiles"`` or ``"all"``) and return either a
Polars DataFrame (``get_summary_stats``) or a formatted string
(``format_summary_stats``).

Refactored from ``unittest.TestCase`` to plain pytest class; uses
``test_data_file`` from conftest.
"""

import polars as pl

from aiqclib.interface.stats import format_summary_stats, get_summary_stats


class TestSummaryStats:
    """Tests for ``get_summary_stats`` and ``format_summary_stats``.

    Coverage:
    - get_summary_stats returns a DataFrame for both ``"profiles"`` and ``"all"`` modes
    - format_summary_stats produces a string with expected variable/stat names
    - The optional ``variables`` and ``summary_stats`` filter parameters work
    """

    def test_get_profile_summary_stats(self, test_data_file):
        """``"profiles"`` mode returns a Polars DataFrame."""
        ds = get_summary_stats(test_data_file, "profiles")
        assert isinstance(ds, pl.DataFrame)

    def test_get_global_summary_stats(self, test_data_file):
        """``"all"`` mode returns a Polars DataFrame."""
        ds = get_summary_stats(test_data_file, "all")
        assert isinstance(ds, pl.DataFrame)

    def test_format_profile_summary_stats(self, test_data_file):
        """``format_summary_stats`` filters by ``variables`` and ``summary_stats``."""
        ds = get_summary_stats(test_data_file, "profiles")

        # Unfiltered: all variables and stats appear.
        stats_str = format_summary_stats(ds)
        assert isinstance(stats_str, str)
        assert "psal" in stats_str
        assert "pct25" in stats_str

        # variables=["pres", "temp"] excludes psal.
        stats_str_filtered_vars = format_summary_stats(ds, ["pres", "temp"])
        assert isinstance(stats_str_filtered_vars, str)
        assert "psal" not in stats_str_filtered_vars
        assert "pct25" in stats_str_filtered_vars

        # summary_stats=["mean"] further restricts to mean only.
        stats_str_filtered_stats = format_summary_stats(ds, ["pres", "temp"], ["mean"])
        assert isinstance(stats_str_filtered_stats, str)
        assert "psal" not in stats_str_filtered_stats
        assert "pct25" not in stats_str_filtered_stats
        assert "mean" in stats_str_filtered_stats

    def test_format_global_summary_stats(self, test_data_file):
        """``format_summary_stats`` on ``"all"`` mode filters the same way."""
        ds = get_summary_stats(test_data_file, "all")

        stats_str = format_summary_stats(ds)
        assert isinstance(stats_str, str)
        assert "psal" in stats_str

        stats_str_filtered_vars = format_summary_stats(ds, ["pres", "temp"])
        assert isinstance(stats_str_filtered_vars, str)
        assert "psal" not in stats_str_filtered_vars