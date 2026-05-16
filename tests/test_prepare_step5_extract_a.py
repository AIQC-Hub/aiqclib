"""Unit tests for the ``ExtractDataSetA`` class.

ExtractDataSetA gathers outputs from steps 1–4 (input, summary, select,
locate) and produces per-target feature DataFrames keyed by target name.

Refactored from three ``unittest.TestCase``/pytest classes
(``TestExtractDataSetA`` over configs 001+004, ``TestExtractDataSetANegX5``
for config 003, ``TestExtractDataSetAwithAll`` for config 005) into three
plain pytest classes that share ``build_prepare_pipeline()`` from conftest.
Per-target triplication collapses to ``for tgt in TARGETS:`` loops.
"""

import os

import polars as pl
import pytest

from aiqclib.prepare.step5_extract_features.dataset_a import ExtractDataSetA

from tests.conftest import TARGETS, build_prepare_pipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_extract(pipeline) -> ExtractDataSetA:
    """Construct an ExtractDataSetA from a build_prepare_pipeline result.

    The four upstream stages (input, summary, select, locate) all live on
    the SimpleNamespace returned by build_prepare_pipeline(..., stop_after="locate").
    """
    return ExtractDataSetA(
        pipeline.config,
        input_data=pipeline.input.input_data,
        selected_profiles=pipeline.select.selected_profiles,
        selected_rows=pipeline.locate.selected_rows,
        summary_stats=pipeline.summary.summary_stats,
    )


# ---------------------------------------------------------------------------
# Tests against configs 001 + 004
# ---------------------------------------------------------------------------

class TestExtractDataSetA:
    """Tests parametrized over configs 001 and 004.

    Both configs use the default selection ratio. Each test runs against
    both via ``@pytest.mark.parametrize("idx", range(2))``.
    """

    @pytest.fixture
    def pipelines(self, dataset_config_001, dataset_config_004, test_data_file):
        """Two pipelines, one per config, each run through step4 (locate)."""
        return [
            build_prepare_pipeline(c, test_data_file, stop_after="locate")
            for c in [dataset_config_001, dataset_config_004]
        ]

    @pytest.mark.parametrize("idx", range(2))
    def test_output_file_names(self, idx, pipelines):
        """Default per-target output paths derive from config.path_info."""
        ds = ExtractDataSetA(pipelines[idx].config)
        base = "/path/to/data_1/nrt_bo_001/extract"
        for tgt in TARGETS:
            assert (
                str(ds.output_file_names[tgt])
                == f"{base}/extracted_features_{tgt}.parquet"
            )

    @pytest.mark.parametrize("idx", range(2))
    def test_step_name(self, idx, pipelines):
        """step_name == 'extract'."""
        ds = ExtractDataSetA(pipelines[idx].config)
        assert ds.step_name == "extract"

    @pytest.mark.parametrize("idx", range(2))
    def test_init_arguments(self, idx, pipelines):
        """All four upstream inputs land on the ExtractDataSetA instance."""
        ds = _make_extract(pipelines[idx])

        assert isinstance(ds.input_data, pl.DataFrame)
        assert ds.input_data.shape[0] == 3267
        assert ds.input_data.shape[1] == 30

        assert isinstance(ds.summary_stats, pl.DataFrame)
        assert ds.summary_stats.shape[0] == 65
        assert ds.summary_stats.shape[1] == 12

        assert isinstance(ds.selected_profiles, pl.DataFrame)
        assert ds.selected_profiles.shape[0] == 14
        assert ds.selected_profiles.shape[1] == 8

        assert isinstance(ds.filtered_input, pl.DataFrame)
        assert ds.filtered_input.shape[0] == 2879
        assert ds.filtered_input.shape[1] == 30

        expected_rows = {"temp": 24, "psal": 36, "pres": 18}
        for tgt in TARGETS:
            assert isinstance(ds.selected_rows[tgt], pl.DataFrame)
            assert ds.selected_rows[tgt].shape[0] == expected_rows[tgt]
            assert ds.selected_rows[tgt].shape[1] == 9

    @pytest.mark.parametrize("idx", range(2))
    def test_location_features(self, idx, pipelines):
        """process_targets populates target_features per target."""
        ds = _make_extract(pipelines[idx])
        ds.process_targets()

        expected_rows = {"temp": 24, "psal": 36, "pres": 18}
        for tgt in TARGETS:
            assert isinstance(ds.target_features[tgt], pl.DataFrame)
            assert ds.target_features[tgt].shape[0] == expected_rows[tgt]
            assert ds.target_features[tgt].shape[1] == 58

    @pytest.mark.parametrize("idx", range(2))
    def test_write_target_features(self, idx, pipelines, test_output_dir):
        """write_target_features produces a parquet per target."""
        ds = _make_extract(pipelines[idx])
        output_paths = {
            tgt: str(test_output_dir / f"test_extracted_features_{tgt}.parquet")
            for tgt in TARGETS
        }
        for tgt in TARGETS:
            ds.output_file_names[tgt] = output_paths[tgt]

        ds.process_targets()
        ds.write_target_features()

        for tgt in TARGETS:
            assert os.path.exists(output_paths[tgt])
            os.remove(output_paths[tgt])  # comment out to debug

    @pytest.mark.parametrize("idx", range(2))
    def test_write_no_target_features(self, idx, pipelines):
        """write_target_features before process_targets raises ValueError."""
        ds = _make_extract(pipelines[idx])
        with pytest.raises(ValueError):
            ds.write_target_features()


# ---------------------------------------------------------------------------
# Tests against config 003 (NegX5)
# ---------------------------------------------------------------------------

class TestExtractDataSetANegX5:
    """Tests against the NegX5 variant (config 003, neg_x_multiplier active)."""

    @pytest.fixture
    def pipeline(self, dataset_config_003, test_data_file):
        return build_prepare_pipeline(
            dataset_config_003, test_data_file, stop_after="locate",
        )

    def test_init_arguments(self, pipeline):
        """All four upstream inputs are present with NegX5-scaled dimensions."""
        ds = _make_extract(pipeline)

        assert isinstance(ds.input_data, pl.DataFrame)
        assert ds.input_data.shape[0] == 3267
        assert ds.input_data.shape[1] == 30

        assert isinstance(ds.summary_stats, pl.DataFrame)
        assert ds.summary_stats.shape[0] == 65
        assert ds.summary_stats.shape[1] == 12

        assert isinstance(ds.selected_profiles, pl.DataFrame)
        assert ds.selected_profiles.shape[0] == 42
        assert ds.selected_profiles.shape[1] == 8

        assert isinstance(ds.filtered_input, pl.DataFrame)
        assert ds.filtered_input.shape[0] == 3267
        assert ds.filtered_input.shape[1] == 30

        expected_rows = {"temp": 177, "psal": 249, "pres": 129}
        for tgt in TARGETS:
            assert isinstance(ds.selected_rows[tgt], pl.DataFrame)
            assert ds.selected_rows[tgt].shape[0] == expected_rows[tgt]
            assert ds.selected_rows[tgt].shape[1] == 9

    def test_location_features(self, pipeline):
        """NegX5: process_targets produces the multiplied-negative feature counts."""
        ds = _make_extract(pipeline)
        ds.process_targets()

        expected_rows = {"temp": 177, "psal": 249, "pres": 129}
        for tgt in TARGETS:
            assert isinstance(ds.target_features[tgt], pl.DataFrame)
            assert ds.target_features[tgt].shape[0] == expected_rows[tgt]
            assert ds.target_features[tgt].shape[1] == 58

    def test_write_target_features(self, pipeline, test_output_dir):
        """write_target_features (NegX5) produces a parquet per target."""
        ds = _make_extract(pipeline)
        output_paths = {
            tgt: str(test_output_dir / f"test_extracted_features_negx5_{tgt}.parquet")
            for tgt in TARGETS
        }
        for tgt in TARGETS:
            ds.output_file_names[tgt] = output_paths[tgt]

        ds.process_targets()
        ds.write_target_features()

        for tgt in TARGETS:
            assert os.path.exists(output_paths[tgt])
            os.remove(output_paths[tgt])  # comment out to debug


# ---------------------------------------------------------------------------
# Tests against config 005 (selects all profiles)
# ---------------------------------------------------------------------------

class TestExtractDataSetAwithAll:
    """Tests against config 005, which selects all profiles (no filtering).

    With no profile filtering, ``filtered_input`` is the full input data,
    ``selected_rows`` per target equals the full input row count, and
    ``target_features`` rows match the input.
    """

    @pytest.fixture
    def pipeline(self, dataset_config_005, test_data_file):
        return build_prepare_pipeline(
            dataset_config_005, test_data_file, stop_after="locate",
        )

    def test_init_arguments(self, pipeline):
        """With select-all config, filtered_input and selected_rows equal full input."""
        ds = _make_extract(pipeline)

        assert isinstance(ds.input_data, pl.DataFrame)
        assert ds.input_data.shape[0] == 3267
        assert ds.input_data.shape[1] == 30

        assert isinstance(ds.summary_stats, pl.DataFrame)
        assert ds.summary_stats.shape[0] == 65
        assert ds.summary_stats.shape[1] == 12

        assert isinstance(ds.selected_profiles, pl.DataFrame)
        # TODO: update to actual value after data reduction (was 503 × 8)
        assert ds.selected_profiles.shape[0] == 12
        assert ds.selected_profiles.shape[1] == 8

        # filtered_input == input_data when all profiles are selected.
        assert isinstance(ds.filtered_input, pl.DataFrame)
        assert ds.filtered_input.shape[0] == 3267
        assert ds.filtered_input.shape[1] == 30

        # All three targets have the full input as their selected_rows.
        for tgt in TARGETS:
            assert isinstance(ds.selected_rows[tgt], pl.DataFrame)
            assert ds.selected_rows[tgt].shape[0] == 3267
            assert ds.selected_rows[tgt].shape[1] == 9

    def test_location_features(self, pipeline):
        """With select-all config, target_features rows match input rows."""
        ds = _make_extract(pipeline)
        ds.process_targets()

        for tgt in TARGETS:
            assert isinstance(ds.target_features[tgt], pl.DataFrame)
            assert ds.target_features[tgt].shape[0] == 3267
            assert ds.target_features[tgt].shape[1] == 58

    def test_write_target_features(self, pipeline, test_output_dir):
        """write_target_features (select-all variant) produces a parquet per target."""
        ds = _make_extract(pipeline)
        output_paths = {
            tgt: str(test_output_dir / f"test_extracted_features_all_{tgt}.parquet")
            for tgt in TARGETS
        }
        for tgt in TARGETS:
            ds.output_file_names[tgt] = output_paths[tgt]

        ds.process_targets()
        ds.write_target_features()

        for tgt in TARGETS:
            assert os.path.exists(output_paths[tgt])
            os.remove(output_paths[tgt])  # comment out to debug