"""Unit tests for the QC flag constants and helpers (``common.utils.qc_flags``)."""

import polars as pl

from aiqclib.common.utils.qc_flags import (
    FLAG_BAD,
    FLAG_GOOD,
    FLAG_PROBABLY_BAD,
    FLAG_SEVERITY_ORDER,
    worst_flag,
)


class TestFlagConstants:
    """The IOC/Argo flag subset and its severity order."""

    def test_values(self):
        """Constants match the flag scheme (1 good, 3 probably bad, 4 bad)."""
        assert FLAG_GOOD == 1
        assert FLAG_PROBABLY_BAD == 3
        assert FLAG_BAD == 4

    def test_severity_is_numeric_order(self):
        """Severity order is ascending numerically (worst_flag relies on it)."""
        assert FLAG_SEVERITY_ORDER == (1, 3, 4)
        assert list(FLAG_SEVERITY_ORDER) == sorted(FLAG_SEVERITY_ORDER)


class TestWorstFlag:
    """Element-wise aggregation of per-item flag columns."""

    def test_worst_across_columns(self):
        """The most severe flag wins per row."""
        df = pl.DataFrame(
            {
                "a": [1, 1, 3, 4],
                "b": [1, 3, 1, 1],
                "c": [1, 1, 4, 3],
            }
        )
        result = df.select(worst_flag("a", "b", "c").alias("flag"))["flag"]
        assert result.to_list() == [1, 3, 4, 4]

    def test_accepts_expressions(self):
        """Expressions and column names can be mixed."""
        df = pl.DataFrame({"a": [1, 4], "b": [3, 1]})
        result = df.select(worst_flag(pl.col("a"), "b").alias("flag"))["flag"]
        assert result.to_list() == [3, 4]

    def test_nulls_ignored(self):
        """A null flag (item not applicable) cannot degrade the result."""
        df = pl.DataFrame({"a": [1, None], "b": [None, 3]})
        result = df.select(worst_flag("a", "b").alias("flag"))["flag"]
        assert result.to_list() == [1, 3]

    def test_single_column(self):
        """A single column passes through unchanged."""
        df = pl.DataFrame({"a": [1, 3, 4]})
        result = df.select(worst_flag("a").alias("flag"))["flag"]
        assert result.to_list() == [1, 3, 4]
