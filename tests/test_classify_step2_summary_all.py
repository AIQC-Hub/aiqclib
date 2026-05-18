"""Unit tests for the ``SummaryDataSetAll`` class.

SummaryDataSetAll is the classify-side analogue of SummaryDataSetA: it
computes per-target global and profile-level statistics over the input
data. Tests are parametrized over two classify configs (001 + 002) since
both should behave identically.

Refactored from a pytest-style class (already pytest, not unittest) with
ad-hoc ``_setup_configs`` / ``_setup_input_datasets`` helpers, replaced
by ``classify_input_001`` / ``classify_input_002`` fixtures from conftest.
"""

import os
from types import SimpleNamespace

import polars as pl
import pytest

from aiqclib.classify.step2_calc_stats.dataset_all import SummaryDataSetAll


# ---------------------------------------------------------------------------
# Fixture: pair of (config, input) for the two-config parametrize
# ---------------------------------------------------------------------------

@pytest.fixture
def step2_setup(
    classify_config_001, classify_config_002,
    classify_input_001, classify_input_002,
):
    """SimpleNamespace combining the two configs and their step1 inputs.

    Tests parametrize over ``idx ∈ {0, 1}`` and index into ``configs`` and
    ``input_ds``. Mirrors the legacy ``self.configs`` / ``self.input_ds``
    lists.
    """
    return SimpleNamespace(
        configs=[classify_config_001, classify_config_002],
        input_ds=[classify_input_001, classify_input_002],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSummaryDataSetAll:
    """Tests for SummaryDataSetAll's stats calculation and file output."""

    @pytest.mark.parametrize("idx", range(2))
    def test_output_file_name(self, idx, step2_setup):
        """Both configs resolve the same configured output path."""
        ds = SummaryDataSetAll(step2_setup.configs[idx])
        assert (
            str(ds.output_file_name)
            == "/path/to/data_1/nrt_bo_001/summary/summary_stats_classify.tsv"
        )

    @pytest.mark.parametrize("idx", range(2))
    def test_step_name(self, idx, step2_setup):
        """step_name == 'summary'."""
        ds = SummaryDataSetAll(step2_setup.configs[idx])
        assert ds.step_name == "summary"

    @pytest.mark.parametrize("idx", range(2))
    def test_input_data(self, idx, step2_setup):
        """input_data is stored as a Polars DataFrame with the expected shape."""
        ds = SummaryDataSetAll(
            step2_setup.configs[idx], input_data=step2_setup.input_ds[idx].input_data,
        )
        assert isinstance(ds.input_data, pl.DataFrame)
        assert ds.input_data.shape[0] == 2456
        assert ds.input_data.shape[1] == 30

    @pytest.mark.parametrize("idx", range(2))
    def test_global_stats(self, idx, step2_setup):
        """calculate_global_stats returns a 1-row, 12-column DataFrame per target."""
        ds = SummaryDataSetAll(
            step2_setup.configs[idx], input_data=step2_setup.input_ds[idx].input_data,
        )
        df = ds.calculate_global_stats("temp")
        assert isinstance(df, pl.DataFrame)
        # 1 row × 12 cols is structural (one row of aggregate stats); not data-dependent.
        assert df.shape[0] == 1
        assert df.shape[1] == 12

    @pytest.mark.parametrize("idx", range(2))
    def test_profile_stats(self, idx, step2_setup):
        """calculate_profile_stats returns one row per profile-group."""
        ds = SummaryDataSetAll(
            step2_setup.configs[idx], input_data=step2_setup.input_ds[idx].input_data,
        )
        grouped_df = ds.input_data.group_by(ds.profile_col_names)
        df = ds.calculate_profile_stats(grouped_df, "temp")
        assert df.shape[0] == 10
        assert df.shape[1] == 12

    @pytest.mark.parametrize("idx", range(2))
    def test_summary_stats(self, idx, step2_setup):
        """calculate_stats populates summary_stats with the full aggregated frame."""
        ds = SummaryDataSetAll(
            step2_setup.configs[idx], input_data=step2_setup.input_ds[idx].input_data,
        )
        ds.calculate_stats()
        assert ds.summary_stats.shape[0] == 55
        assert ds.summary_stats.shape[1] == 12

    @pytest.mark.parametrize("idx", range(2))
    def test_write_summary_stats(self, idx, step2_setup, test_output_dir):
        """write_summary_stats produces a TSV at the configured path."""
        ds = SummaryDataSetAll(
            step2_setup.configs[idx], input_data=step2_setup.input_ds[idx].input_data,
        )
        output_path = str(test_output_dir / "test_summary_stats_classify.tsv")
        ds.output_file_name = output_path

        ds.calculate_stats()
        ds.write_summary_stats()
        assert os.path.exists(output_path)
        os.remove(output_path)  # comment out to debug

    @pytest.mark.parametrize("idx", range(2))
    def test_write_no_summary_stats(self, idx, step2_setup):
        """write_summary_stats before calculate_stats raises ValueError."""
        ds = SummaryDataSetAll(
            step2_setup.configs[idx], input_data=step2_setup.input_ds[idx].input_data,
        )
        with pytest.raises(ValueError):
            ds.write_summary_stats()