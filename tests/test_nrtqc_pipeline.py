"""Integration test for the NRT QC pipeline (steps 1-3) on the CTD fixture.

Runs read input -> run QC items -> aggregate flags end to end with the
template configuration (Mediterranean regional ranges against Baltic
fixture data, so the regional range genuinely fires) and checks the final
output frame and file.

Also covers ``include_in_final_flag: false`` on real data: the column set
is unchanged and the excluded item's column is byte-identical, while the
final flag drops it.
"""

import os

import polars as pl
import pytest

from aiqclib.common.loader.nrtqc_loader import (
    load_nrtqc_step1_input_dataset,
    load_nrtqc_step2_qc_dataset,
    load_nrtqc_step3_concat_dataset,
)
from aiqclib.common.utils.qc_flags import worst_flag


@pytest.fixture
def pipeline(nrtqc_config_001):
    """The three step objects after a full pipeline run."""
    ds_input = load_nrtqc_step1_input_dataset(nrtqc_config_001)
    ds_input.read_input_data()

    ds_qc = load_nrtqc_step2_qc_dataset(nrtqc_config_001, ds_input.input_data)
    ds_qc.run_qc_items()

    ds_concat = load_nrtqc_step3_concat_dataset(nrtqc_config_001, ds_qc.qc_data)
    ds_concat.concat_flags()
    return ds_input, ds_qc, ds_concat


class TestNRTQCPipeline:
    """End-to-end behaviour of steps 1-3."""

    def test_output_shape_and_columns(self, pipeline):
        """Every input row and column survives; QC columns are added."""
        ds_input, ds_qc, ds_concat = pipeline
        merged = ds_concat.merged_data

        assert merged.shape[0] == ds_input.input_data.shape[0] == 3267
        assert set(ds_input.input_data.columns) <= set(merged.columns)
        # 17 item columns + propagation column + 2 final flags.
        assert merged.shape[1] == ds_input.input_data.shape[1] + 20
        assert {"temp_nrt_flag", "psal_nrt_flag", "psal_qc_temp_to_psal"} <= set(
            merged.columns
        )

    def test_final_flags_never_null_and_in_scheme(self, pipeline):
        _, _, ds_concat = pipeline
        for column in ("temp_nrt_flag", "psal_nrt_flag"):
            flags = ds_concat.merged_data[column]
            assert flags.null_count() == 0
            assert set(flags.unique().to_list()) <= {1, 3, 4}

    def test_some_observations_flagged(self, pipeline):
        """Mediterranean ranges against Baltic data must flag something."""
        _, _, ds_concat = pipeline
        temp_flags = ds_concat.merged_data["temp_nrt_flag"].to_list()
        assert temp_flags.count(4) > 0
        assert temp_flags.count(1) > 0

    def test_temp_flag_consistent_with_item_columns(self, pipeline):
        """temp_nrt_flag equals the worst of temp's applicable columns."""
        _, ds_qc, ds_concat = pipeline
        merged = ds_concat.merged_data

        applicable = [
            c
            for c in ds_qc.qc_item_columns()
            if c.startswith("temp_qc_") or c.startswith("qc_")
        ]
        recomputed = merged.select(worst_flag(*applicable).alias("expected"))[
            "expected"
        ]
        assert (merged["temp_nrt_flag"] == recomputed).all()

    def test_psal_flag_includes_propagation(self, pipeline):
        """psal_nrt_flag is at least as severe as the propagated temp flag."""
        _, _, ds_concat = pipeline
        merged = ds_concat.merged_data
        assert (merged["psal_nrt_flag"] >= merged["psal_qc_temp_to_psal"]).all()

    def test_write_output(self, pipeline):
        _, _, ds_concat = pipeline
        ds_concat.write_merged_data()

        assert os.path.exists(ds_concat.output_file_name)
        written = pl.read_parquet(ds_concat.output_file_name)
        assert written.shape == ds_concat.merged_data.shape
        os.remove(ds_concat.output_file_name)


class TestFinalFlagItemSelectionEndToEnd:
    """Excluding an item end to end on the fixture, not a synthetic frame.

    The regional range is the item to test with here: the Mediterranean
    template ranges against Baltic fixture data make it fire on real rows,
    so excluding it has to change the final flag on real data.
    """

    @pytest.fixture
    def excluded_pipeline(self, nrtqc_config_001):
        """The same pipeline with regional_range out of the final flag."""
        by_name = {x["name"]: x for x in nrtqc_config_001.data["qc_item_set"]["items"]}
        by_name["regional_range"]["include_in_final_flag"] = False

        ds_input = load_nrtqc_step1_input_dataset(nrtqc_config_001)
        ds_input.read_input_data()
        ds_qc = load_nrtqc_step2_qc_dataset(nrtqc_config_001, ds_input.input_data)
        ds_qc.run_qc_items()
        ds_concat = load_nrtqc_step3_concat_dataset(nrtqc_config_001, ds_qc.qc_data)
        ds_concat.concat_flags()
        return ds_input, ds_qc, ds_concat

    def test_column_set_is_unchanged(self, pipeline, excluded_pipeline):
        """The output keeps exactly the same columns and rows."""
        _, _, baseline = pipeline
        _, _, excluded = excluded_pipeline

        assert excluded.merged_data.columns == baseline.merged_data.columns
        assert excluded.merged_data.shape == baseline.merged_data.shape

    def test_excluded_item_column_is_identical(self, pipeline, excluded_pipeline):
        """The item still runs: its column matches the baseline exactly."""
        _, _, baseline = pipeline
        _, _, excluded = excluded_pipeline

        column = "temp_qc_regional_range"
        assert (
            excluded.merged_data[column].to_list()
            == baseline.merged_data[column].to_list()
        )
        # And it genuinely fires, so the comparison below means something.
        assert excluded.merged_data[column].to_list().count(4) > 0

    def test_final_flag_no_longer_reflects_the_item(self, excluded_pipeline):
        """temp_nrt_flag now equals the worst of the remaining columns."""
        _, ds_qc, ds_concat = excluded_pipeline
        merged = ds_concat.merged_data

        remaining = [
            c
            for c in ds_qc.qc_item_columns()
            if (c.startswith("temp_qc_") or c.startswith("qc_"))
            and not c.endswith("_regional_range")
        ]
        recomputed = merged.select(worst_flag(*remaining).alias("expected"))["expected"]
        assert (merged["temp_nrt_flag"] == recomputed).all()

    def test_fewer_rows_flagged(self, pipeline, excluded_pipeline):
        """Dropping an item that fires can only relax the final flag."""
        _, _, baseline = pipeline
        _, _, excluded = excluded_pipeline

        before = baseline.merged_data["temp_nrt_flag"]
        after = excluded.merged_data["temp_nrt_flag"]
        assert (after <= before).all()
        assert after.to_list().count(4) < before.to_list().count(4)
