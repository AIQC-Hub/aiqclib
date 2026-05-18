"""Unit tests for the classify-side ``ExtractDataSetAll`` class.

ExtractDataSetAll is the classify-side analogue of the prepare-side
ExtractDataSetAll: it gathers outputs from classify steps 1-4 (input,
summary, select, locate) and produces per-target feature DataFrames keyed
by target name.

Refactored from a pytest-style class with five ad-hoc helper methods
(``_setup_configs``, ``_setup_input_datasets``, ``_setup_summary_datasets``,
``_setup_select_datasets``, ``_setup_locate_datasets``), all replaced by a
single ``pipelines`` fixture that calls ``build_classify_prepare_pipeline``
from conftest with ``stop_after="locate"``. Per-target triplication
collapses to ``for tgt in TARGETS:`` loops.
"""

import os

import polars as pl
import pytest

from aiqclib.classify.step5_extract_features.dataset_all import ExtractDataSetAll

from tests.conftest import TARGETS, build_classify_prepare_pipeline


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_extract(pipeline) -> ExtractDataSetAll:
    """Construct an ExtractDataSetAll from a build_classify_prepare_pipeline result.

    The four upstream stages (input, summary, select, locate) all live on
    the SimpleNamespace returned by build_classify_prepare_pipeline(
    ..., stop_after="locate").
    """
    return ExtractDataSetAll(
        pipeline.config,
        input_data=pipeline.input.input_data,
        selected_profiles=pipeline.select.selected_profiles,
        selected_rows=pipeline.locate.selected_rows,
        summary_stats=pipeline.summary.summary_stats,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestExtractDataSetAll:
    """Tests for the classify-side ExtractDataSetAll, parametrized over two configs.

    Both configs (001 + 002) should behave identically; tests parametrize
    over ``idx ∈ {0, 1}`` to run each in isolation.
    """

    @pytest.fixture
    def pipelines(self, classify_config_001, classify_config_002, test_data_file):
        """Two pipelines, one per config, each run through step4 (locate)."""
        return [
            build_classify_prepare_pipeline(c, test_data_file, stop_after="locate")
            for c in [classify_config_001, classify_config_002]
        ]

    @pytest.mark.parametrize("idx", range(2))
    def test_output_file_names(self, idx, pipelines):
        """Default per-target output paths derive from config.path_info."""
        ds = ExtractDataSetAll(pipelines[idx].config)
        base = "/path/to/data_1/nrt_bo_001/extract"
        for tgt in TARGETS:
            assert (
                str(ds.output_file_names[tgt])
                == f"{base}/extracted_features_classify_{tgt}.parquet"
            )

    @pytest.mark.parametrize("idx", range(2))
    def test_step_name(self, idx, pipelines):
        """step_name == 'extract'."""
        ds = ExtractDataSetAll(pipelines[idx].config)
        assert ds.step_name == "extract"

    @pytest.mark.parametrize("idx", range(2))
    def test_init_arguments(self, idx, pipelines):
        """All four upstream inputs land on the ExtractDataSetAll instance."""
        ds = _make_extract(pipelines[idx])

        assert isinstance(ds.input_data, pl.DataFrame)
        assert ds.input_data.shape[0] == 2456
        assert ds.input_data.shape[1] == 30

        assert isinstance(ds.summary_stats, pl.DataFrame)
        assert ds.summary_stats.shape[0] == 55
        assert ds.summary_stats.shape[1] == 12

        assert isinstance(ds.selected_profiles, pl.DataFrame)
        assert ds.selected_profiles.shape[0] == 10
        assert ds.selected_profiles.shape[1] == 8

        # Classify-side keeps all input rows per target.
        for tgt in TARGETS:
            assert isinstance(ds.selected_rows[tgt], pl.DataFrame)
            assert ds.selected_rows[tgt].shape[0] == 2456
            assert ds.selected_rows[tgt].shape[1] == 9

    @pytest.mark.parametrize("idx", range(2))
    def test_location_features(self, idx, pipelines):
        """process_targets populates target_features per target.

        Note: the original test method was named ``test_location_features``
        but exercises ``ExtractDataSetAll.process_targets``, not a location-
        specific feature. Preserving the name for git-history continuity;
        ``pytest -k`` filters will still match.
        """
        ds = _make_extract(pipelines[idx])
        ds.process_targets()

        for tgt in TARGETS:
            assert isinstance(ds.target_features[tgt], pl.DataFrame)
            assert ds.target_features[tgt].shape[0] == 2456
            assert ds.target_features[tgt].shape[1] == 56

    @pytest.mark.parametrize("idx", range(2))
    def test_write_target_features(self, idx, pipelines, test_output_dir):
        """write_target_features produces a parquet per target."""
        ds = _make_extract(pipelines[idx])
        output_paths = {
            tgt: str(test_output_dir / f"test_extracted_features_classify_{tgt}.parquet")
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