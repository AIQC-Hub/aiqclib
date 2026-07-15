"""Unit tests for the NRT QC module's step 4 (flag comparison report).

Uses a synthetic merged frame with known existing and new flags so the
contingency counts, agreement metrics, and item breakdown can be checked
against hand-computed values, plus an integration run on the CTD fixture.
"""

import os

import polars as pl
import pytest
from polars.exceptions import ColumnNotFoundError

from aiqclib.common.loader.nrtqc_loader import (
    load_nrtqc_step1_input_dataset,
    load_nrtqc_step2_qc_dataset,
    load_nrtqc_step3_concat_dataset,
    load_nrtqc_step4_compare_dataset,
)
from aiqclib.nrtqc.step4_compare_flags.dataset_all import CompareFlagsAll


def make_merged_frame(temp_qc_dtype: pl.DataType = pl.Int64) -> pl.DataFrame:
    """A synthetic step-3 output with hand-picked flag disagreements.

    Existing temp flags mark observations 5 and 6 bad; the new flags mark
    observations 3 and 5 bad. So: 5 agree-good, 1 agree-bad (obs 5),
    1 new-only (obs 3), 1 existing-only (obs 6).
    """
    n = 8
    temp_qc = [1, 1, 1, 1, 4, 4, 1, 1]
    if temp_qc_dtype == pl.Utf8:
        temp_qc = [str(x) for x in temp_qc]
    return pl.DataFrame(
        {
            "platform_code": ["P1"] * n,
            "profile_no": [1] * n,
            "observation_no": list(range(1, n + 1)),
            "temp": [5.0] * n,
            "psal": [35.0] * n,
            "temp_qc": pl.Series(temp_qc, dtype=temp_qc_dtype),
            "psal_qc": pl.Series([1] * n, dtype=pl.Int64),
            "temp_nrt_flag": pl.Series([1, 1, 4, 1, 4, 1, 1, 1], dtype=pl.Int64),
            "psal_nrt_flag": pl.Series([1] * n, dtype=pl.Int64),
            "temp_qc_global_range": pl.Series([1, 1, 4, 1, 4, 1, 1, 1], dtype=pl.Int64),
            "qc_impossible_date": pl.Series([1] * n, dtype=pl.Int64),
        }
    )


def get_section(report: pl.DataFrame, section: str) -> pl.DataFrame:
    return report.filter(pl.col("section") == section)


@pytest.fixture
def temp_report(nrtqc_config_001):
    """The temp comparison report for the synthetic frame."""
    ds = load_nrtqc_step4_compare_dataset(nrtqc_config_001, make_merged_frame())
    ds.compare_targets()
    return ds.reports["temp"]


class TestContingency:
    """Section 1: existing x new cross-tabulation."""

    def test_counts_and_percentages(self, temp_report):
        contingency = get_section(temp_report, "contingency")
        rows = [
            (r["existing_flag"], r["new_flag"], r["count"], r["percent"])
            for r in contingency.iter_rows(named=True)
        ]
        assert rows == [
            (1, 1, 5, 62.5),
            (1, 4, 1, 12.5),
            (4, 1, 1, 12.5),
            (4, 4, 1, 12.5),
        ]

    def test_counts_sum_to_total(self, temp_report):
        contingency = get_section(temp_report, "contingency")
        assert contingency["count"].sum() == 8

    def test_string_flags_handled(self, nrtqc_config_001):
        """String-typed existing flags are cast for the comparison."""
        ds = load_nrtqc_step4_compare_dataset(
            nrtqc_config_001, make_merged_frame(temp_qc_dtype=pl.Utf8)
        )
        ds.compare_targets()
        contingency = get_section(ds.reports["temp"], "contingency")
        assert contingency["count"].sum() == 8
        assert set(contingency["existing_flag"].to_list()) == {1, 4}


class TestAgreement:
    """Section 2: binary agreement metrics (pos/neg convention)."""

    def test_metrics(self, temp_report):
        agreement = get_section(temp_report, "agreement")
        metrics = dict(zip(agreement["metric"], agreement["value"]))
        # actual positive: obs 5, 6; predicted positive: obs 3, 5.
        assert metrics["n_used"] == 8.0
        assert metrics["tp"] == 1.0
        assert metrics["fp"] == 1.0
        assert metrics["fn"] == 1.0
        assert metrics["tn"] == 5.0
        assert metrics["accuracy"] == 0.75
        assert metrics["precision"] == 0.5
        assert metrics["recall"] == 0.5

    def test_metrics_only_with_pos_neg_values(self, nrtqc_config_001):
        """Without pos/neg flag values there is no binarisation."""
        variables = nrtqc_config_001.data["qc_variable_set"]["variables"]
        by_name = {v["name"]: v for v in variables}
        by_name["temp"].pop("pos_flag_values")

        ds = load_nrtqc_step4_compare_dataset(nrtqc_config_001, make_merged_frame())
        ds.compare_targets()
        assert get_section(ds.reports["temp"], "agreement").height == 0
        # The contingency table is still produced.
        assert get_section(ds.reports["temp"], "contingency").height > 0


class TestItemBreakdown:
    """Section 3: per-item failure counts by existing flag value."""

    def test_breakdown_counts(self, temp_report):
        breakdown = get_section(temp_report, "item_breakdown").filter(
            pl.col("item") == "global_range"
        )
        rows = {
            r["existing_flag"]: (r["count"], r["percent"])
            for r in breakdown.iter_rows(named=True)
        }
        # Existing-good rows: 6, one flagged by the item (obs 3).
        # Existing-bad rows: 2, one flagged by the item (obs 5).
        assert rows[1] == (1, pytest.approx(100 / 6))
        assert rows[4] == (1, pytest.approx(50.0))

    def test_only_present_columns_reported(self, temp_report):
        """Items without columns in the frame produce no breakdown rows."""
        breakdown = get_section(temp_report, "item_breakdown")
        assert set(breakdown["item"].unique().to_list()) == {
            "global_range",
            "impossible_date",
        }


class TestComparableTargets:
    """Skip/error behaviour for the configured variables."""

    def test_variable_without_flag_skipped(self, nrtqc_config_001):
        variables = nrtqc_config_001.data["qc_variable_set"]["variables"]
        by_name = {v["name"]: v for v in variables}
        by_name["psal"].pop("flag")

        ds = load_nrtqc_step4_compare_dataset(nrtqc_config_001, make_merged_frame())
        ds.compare_targets()
        assert set(ds.reports) == {"temp"}

    def test_missing_flag_column_raises(self, nrtqc_config_001):
        """A configured flag column absent from the data is an error."""
        ds = load_nrtqc_step4_compare_dataset(
            nrtqc_config_001, make_merged_frame().drop("psal_qc")
        )
        with pytest.raises(ColumnNotFoundError, match="psal_qc"):
            ds.compare_targets()

    def test_missing_nrt_flag_column_raises(self, nrtqc_config_001):
        ds = load_nrtqc_step4_compare_dataset(
            nrtqc_config_001, make_merged_frame().drop("temp_nrt_flag")
        )
        with pytest.raises(ColumnNotFoundError, match="temp_nrt_flag"):
            ds.compare_targets()

    def test_compare_without_data_raises(self, nrtqc_config_001):
        ds = load_nrtqc_step4_compare_dataset(nrtqc_config_001, None)
        with pytest.raises(ValueError, match="merged_data"):
            ds.compare_targets()

    def test_loader_returns_configured_class(self, nrtqc_config_001):
        ds = load_nrtqc_step4_compare_dataset(nrtqc_config_001)
        assert isinstance(ds, CompareFlagsAll)


class TestWriteReports:
    """Report writing."""

    def test_write(self, nrtqc_config_001):
        ds = load_nrtqc_step4_compare_dataset(nrtqc_config_001, make_merged_frame())
        ds.compare_targets()
        ds.write_reports()

        for target in ("temp", "psal"):
            path = ds.output_file_names[target]
            assert os.path.exists(path)
            os.remove(path)

    def test_write_with_no_comparable_targets(self, nrtqc_config_001):
        """No configured flags: nothing to write, no error."""
        variables = nrtqc_config_001.data["qc_variable_set"]["variables"]
        for v in variables:
            v.pop("flag", None)

        ds = load_nrtqc_step4_compare_dataset(nrtqc_config_001, make_merged_frame())
        ds.compare_targets()
        ds.write_reports()

        assert ds.reports == {}
        for path in ds.output_file_names.values():
            assert not os.path.exists(path)


class TestCompareOnFixture:
    """Integration: compare after the real steps 1-3 pipeline."""

    def test_full_pipeline_comparison(self, nrtqc_config_001):
        ds_input = load_nrtqc_step1_input_dataset(nrtqc_config_001)
        ds_input.read_input_data()
        ds_qc = load_nrtqc_step2_qc_dataset(nrtqc_config_001, ds_input.input_data)
        ds_qc.run_qc_items()
        ds_concat = load_nrtqc_step3_concat_dataset(nrtqc_config_001, ds_qc.qc_data)
        ds_concat.concat_flags()

        ds_compare = load_nrtqc_step4_compare_dataset(
            nrtqc_config_001, ds_concat.merged_data
        )
        ds_compare.compare_targets()

        assert set(ds_compare.reports) == {"temp", "psal"}
        for report in ds_compare.reports.values():
            contingency = get_section(report, "contingency")
            assert contingency["count"].sum() == 3267
            metrics = get_section(report, "agreement")["metric"].to_list()
            assert "accuracy" in metrics
