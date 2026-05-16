"""Unit tests for the ``SummaryDataSetA`` class.

Exercises:
- Identity (step_name), output-path resolution (default + configured)
- input_data wiring
- The four stats-calculation entry points (``calculate_global_stats``,
  ``calculate_profile_stats``, ``calculate_stats``,
  ``create_summary_stats_observation``, ``create_summary_stats_profile``)
- write_summary_stats file output and its empty-state ValueError
- Empty-state ValueErrors for observation and profile creators

Refactored from the original which:
- Was named ``TestSelectDataSetA`` — a clear copy-paste from step3 that
  tested SummaryDataSetA. Renamed to ``TestSummaryDataSetA`` so
  ``pytest -k Summary`` filters it correctly.
- Used ad-hoc ``_setup_configs()`` and ``_setup_input_datasets()`` helpers
  in an autouse fixture (replaced by ``step2_setup`` fixture pulling
  ``dataset_input_001`` / ``dataset_input_004`` from conftest)
- Was already pytest-style; main wins here are conftest fixture reuse and
  TODO markers on data-dependent values
"""

import os
from types import SimpleNamespace

import polars as pl
import pytest

from aiqclib.common.config.dataset_config import DataSetConfig
from aiqclib.prepare.step2_calc_stats.dataset_a import SummaryDataSetA


# ---------------------------------------------------------------------------
# Fixtures specific to this test file
# ---------------------------------------------------------------------------

@pytest.fixture
def step2_setup(dataset_input_001, dataset_input_004):
    """Two-config setup for tests that should behave identically under both.

    The original ``setup_input`` fixture loaded configs 001 and 004 and
    built an input dataset for each. Tests then parametrize over
    ``idx in range(2)``. SimpleNamespace mirrors the legacy ``self.configs``
    / ``self.input_ds`` lists so test bodies stay readable.
    """
    return SimpleNamespace(
        configs=[dataset_input_001.config, dataset_input_004.config],
        input_ds=[dataset_input_001, dataset_input_004],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSummaryDataSetA:
    """Tests for SummaryDataSetA's stats calculation and file output."""

    @pytest.mark.parametrize("idx", range(2))
    def test_output_file_name(self, idx, step2_setup):
        """Both configs resolve the same configured output path."""
        ds = SummaryDataSetA(step2_setup.configs[idx])
        assert (
            str(ds.output_file_name)
            == "/path/to/data_1/nrt_bo_001/summary/summary_stats.tsv"
        )

    def test_default_output_file_name(self, dataset_config_002):
        """test_dataset_002.yaml uses the default-path pattern (no output_file_name set)."""
        ds = SummaryDataSetA(dataset_config_002)
        assert (
            str(ds.output_file_name)
            == "/path/to/data_1/summary_dataset_folder/summary/summary_in_params.txt"
        )

    @pytest.mark.parametrize("idx", range(2))
    def test_step_name(self, idx, step2_setup):
        """step_name == 'summary'."""
        ds = SummaryDataSetA(step2_setup.configs[idx])
        assert ds.step_name == "summary"

    @pytest.mark.parametrize("idx", range(2))
    def test_input_data(self, idx, step2_setup):
        """input_data is stored as a Polars DataFrame with the expected shape."""
        ds = SummaryDataSetA(
            step2_setup.configs[idx], input_data=step2_setup.input_ds[idx].input_data
        )
        assert isinstance(ds.input_data, pl.DataFrame)
        assert ds.input_data.shape[0] == 3267
        assert ds.input_data.shape[1] == 30

    @pytest.mark.parametrize("idx", range(2))
    def test_global_stats(self, idx, step2_setup):
        """calculate_global_stats returns a 1-row, 12-column DataFrame per target."""
        ds = SummaryDataSetA(
            step2_setup.configs[idx], input_data=step2_setup.input_ds[idx].input_data
        )
        df = ds.calculate_global_stats("temp")
        assert isinstance(df, pl.DataFrame)
        # 1 row × 12 cols is structural (one row of aggregate stats); not data-dependent.
        assert df.shape[0] == 1
        assert df.shape[1] == 12

    @pytest.mark.parametrize("idx", range(2))
    def test_profile_stats(self, idx, step2_setup):
        """calculate_profile_stats returns one row per profile-group."""
        ds = SummaryDataSetA(
            step2_setup.configs[idx], input_data=step2_setup.input_ds[idx].input_data
        )
        grouped_df = ds.input_data.group_by(ds.profile_col_names)
        df = ds.calculate_profile_stats(grouped_df, "temp")
        assert df.shape[0] == 12
        assert df.shape[1] == 12

    @pytest.mark.parametrize("idx", range(2))
    def test_summary_stats(self, idx, step2_setup):
        """calculate_stats populates summary_stats with the full aggregated frame."""
        ds = SummaryDataSetA(
            step2_setup.configs[idx], input_data=step2_setup.input_ds[idx].input_data
        )
        ds.calculate_stats()
        assert ds.summary_stats.shape[0] == 65
        assert ds.summary_stats.shape[1] == 12

    @pytest.mark.parametrize("idx", range(2))
    def test_write_summary_stats(self, idx, step2_setup, test_output_dir):
        """write_summary_stats produces a TSV at the configured path."""
        ds = SummaryDataSetA(
            step2_setup.configs[idx], input_data=step2_setup.input_ds[idx].input_data
        )
        output_path = str(test_output_dir / "test_summary_stats.tsv")
        ds.output_file_name = output_path

        ds.calculate_stats()
        ds.write_summary_stats()
        assert os.path.exists(output_path)
        os.remove(output_path)  # comment out to debug

    @pytest.mark.parametrize("idx", range(2))
    def test_write_no_summary_stats(self, idx, step2_setup):
        """write_summary_stats before calculate_stats raises ValueError."""
        ds = SummaryDataSetA(
            step2_setup.configs[idx], input_data=step2_setup.input_ds[idx].input_data
        )
        with pytest.raises(ValueError):
            ds.write_summary_stats()

    @pytest.mark.parametrize("idx", range(2))
    def test_summary_stats_observation(self, idx, step2_setup):
        """create_summary_stats_observation produces a 5×5 observation-level frame."""
        ds = SummaryDataSetA(
            step2_setup.configs[idx], input_data=step2_setup.input_ds[idx].input_data
        )
        ds.calculate_stats()
        ds.create_summary_stats_observation()
        # 5×5 is structural (5 stats × 5 metrics); not data-dependent.
        assert ds.summary_stats_observation.shape[0] == 5
        assert ds.summary_stats_observation.shape[1] == 5

    @pytest.mark.parametrize("idx", range(2))
    def test_summary_stats_observation_without_stats_ds(self, idx, step2_setup):
        """create_summary_stats_observation before calculate_stats raises ValueError."""
        ds = SummaryDataSetA(
            step2_setup.configs[idx], input_data=step2_setup.input_ds[idx].input_data
        )
        with pytest.raises(ValueError):
            ds.create_summary_stats_observation()

    @pytest.mark.parametrize("idx", range(2))
    def test_summary_stats_profile(self, idx, step2_setup):
        """create_summary_stats_profile produces a 27×6 profile-level frame."""
        ds = SummaryDataSetA(
            step2_setup.configs[idx], input_data=step2_setup.input_ds[idx].input_data
        )
        ds.calculate_stats()
        ds.create_summary_stats_profile()
        # TODO: update to actual value after data reduction (was 27 × 6).
        # The "27" might still be 27 (depends on # profiles × stat-types).
        assert ds.summary_stats_profile.shape[0] == 27
        assert ds.summary_stats_profile.shape[1] == 6

    @pytest.mark.parametrize("idx", range(2))
    def test_summary_stats_profile_without_stats_ds(self, idx, step2_setup):
        """create_summary_stats_profile before calculate_stats raises ValueError."""
        ds = SummaryDataSetA(
            step2_setup.configs[idx], input_data=step2_setup.input_ds[idx].input_data
        )
        with pytest.raises(ValueError):
            ds.create_summary_stats_profile()