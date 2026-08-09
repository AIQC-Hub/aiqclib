"""Unit tests for the QC flag constants and helpers (``common.utils.qc_flags``)."""

import polars as pl
import pytest

from aiqclib.common.utils.qc_flags import (
    FLAG_BAD,
    FLAG_GOOD,
    FLAG_PROBABLY_BAD,
    FLAG_SEVERITY_ORDER,
    flag_as_int,
    flag_is_in,
    normalize_flag_values,
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


class TestFlagAsInt:
    """Reading an existing QC flag column whatever dtype it carries."""

    @pytest.mark.parametrize(
        "values, dtype",
        [
            ([1, 4, 9], pl.Int32),
            ([1, 4, 9], pl.Int64),
            ([1, 4, 9], pl.UInt32),
            ([1.0, 4.0, 9.0], pl.Float64),
            (["1", "4", "9"], pl.Utf8),
        ],
        ids=["int32", "int64", "uint32", "float64", "string"],
    )
    def test_reads_every_dtype(self, values, dtype):
        """Integer, float and string flag columns all read as the same Int64."""
        df = pl.DataFrame({"qc": values}, schema={"qc": dtype})
        result = df.select(flag_as_int("qc").alias("f"))["f"]
        assert result.dtype == pl.Int64
        assert result.to_list() == [1, 4, 9]

    def test_unparseable_values_become_null(self):
        """Missing markers and non-numeric codes become null, never a match."""
        df = pl.DataFrame({"qc": ["1", "", "x", None]})
        result = df.select(flag_as_int("qc").alias("f"))["f"]
        assert result.to_list() == [1, None, None, None]

    def test_float_nan_becomes_null(self):
        """NaN, the float spelling of a missing flag, becomes null too."""
        df = pl.DataFrame({"qc": [1.0, float("nan"), None]})
        result = df.select(flag_as_int("qc").alias("f"))["f"]
        assert result.to_list() == [1, None, None]

    def test_string_float_spelling(self):
        """Flags read from CSV as '4.0' parse to the same value as '4'."""
        df = pl.DataFrame({"qc": ["4", "4.0"]})
        result = df.select(flag_as_int("qc").alias("f"))["f"]
        assert result.to_list() == [4, 4]


class TestNormalizeFlagValues:
    """Configured flag values reduced to the Int64 domain."""

    def test_integers_pass_through(self):
        """The usual [4, 6, 7] form is unchanged."""
        assert normalize_flag_values([4, 6, 7]) == [4, 6, 7]

    def test_strings_are_parsed(self):
        """A config written for a string flag column yields the same list."""
        assert normalize_flag_values(["4", "6", "7"]) == [4, 6, 7]

    def test_mixed_values(self):
        """Integers and strings may be mixed in one list."""
        assert normalize_flag_values([1, "3", 4.0]) == [1, 3, 4]

    def test_empty(self):
        """An empty list normalizes to an empty list."""
        assert normalize_flag_values([]) == []

    @pytest.mark.parametrize("value", ["bad", None, "4.5", 4.5])
    def test_non_whole_numbers_raise(self, value):
        """A value that is not a whole number is a configuration error."""
        with pytest.raises(ValueError, match="whole number"):
            normalize_flag_values([value])


class TestFlagIsIn:
    """Membership tests across every column dtype / config value combination."""

    @pytest.mark.parametrize(
        "column_values, dtype",
        [([1, 4, 9], pl.Int32), (["1", "4", "9"], pl.Utf8)],
        ids=["integer_column", "string_column"],
    )
    @pytest.mark.parametrize(
        "flag_values", [[4, 9], ["4", "9"]], ids=["integer_config", "string_config"]
    )
    def test_all_combinations_agree(self, column_values, dtype, flag_values):
        """String and integer spellings match on both sides, in any mix."""
        df = pl.DataFrame({"qc": column_values}, schema={"qc": dtype})
        result = df.select(flag_is_in("qc", flag_values).alias("hit"))["hit"]
        assert result.to_list() == [False, True, True]

    def test_unparseable_flags_do_not_match(self):
        """An empty or non-numeric flag matches neither list."""
        df = pl.DataFrame({"qc": ["", "x", "1"]})
        result = df.select(flag_is_in("qc", [1]).alias("hit"))["hit"]
        assert result.to_list() == [False, False, True]
