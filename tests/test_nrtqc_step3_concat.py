"""Unit tests for the NRT QC module's step 3 (flag aggregation).

Uses small synthetic flag frames so every final NRT flag can be verified
against hand-computed worst-flag expectations, including the deferred
temp-to-psal propagation and its traceable column.
"""

import os

import polars as pl
import pytest

from aiqclib.common.loader.nrtqc_loader import load_nrtqc_step3_concat_dataset


def make_flag_frame(**flag_columns) -> pl.DataFrame:
    """A two-observation frame with keys plus the given flag columns."""
    n = 2
    data = {
        "platform_code": ["P1"] * n,
        "profile_no": [1] * n,
        "observation_no": [1, 2],
        "temp": [5.0, 6.0],
        "psal": [35.0, 35.1],
    }
    for name, values in flag_columns.items():
        data[name] = pl.Series(values, dtype=pl.Int64)
    return pl.DataFrame(data)


class TestConcatFlags:
    """Final NRT flags are the worst applicable item flags."""

    def test_worst_flag_per_variable(self, nrtqc_config_001):
        """Variable columns and profile-level columns both count."""
        df = make_flag_frame(
            qc_impossible_date=[1, 1],
            temp_qc_global_range=[4, 1],
            temp_qc_spike=[1, 3],
            psal_qc_global_range=[1, 1],
        )
        ds = load_nrtqc_step3_concat_dataset(nrtqc_config_001, df)
        ds.concat_flags()

        merged = ds.merged_data
        assert merged["temp_nrt_flag"].to_list() == [4, 3]
        # psal's own items are clean, but temp 4/3 propagates onto psal.
        assert merged["psal_qc_temp_to_psal"].to_list() == [4, 3]
        assert merged["psal_nrt_flag"].to_list() == [4, 3]

    def test_profile_level_flag_applies_to_all_variables(self, nrtqc_config_001):
        df = make_flag_frame(qc_impossible_date=[4, 1])
        ds = load_nrtqc_step3_concat_dataset(nrtqc_config_001, df)
        ds.concat_flags()

        assert ds.merged_data["temp_nrt_flag"].to_list() == [4, 1]
        assert ds.merged_data["psal_nrt_flag"].to_list() == [4, 1]

    def test_no_item_columns_means_good(self, nrtqc_config_001):
        """Without any applicable item columns the final flags are good."""
        df = make_flag_frame()
        ds = load_nrtqc_step3_concat_dataset(nrtqc_config_001, df)
        ds.concat_flags()

        assert ds.merged_data["temp_nrt_flag"].to_list() == [1, 1]
        assert ds.merged_data["psal_nrt_flag"].to_list() == [1, 1]

    def test_propagation_disabled_when_item_omitted(self, nrtqc_config_001):
        """Without temp_to_psal in the item set, psal keeps its own flags."""
        items = nrtqc_config_001.data["qc_item_set"]["items"]
        nrtqc_config_001.data["qc_item_set"]["items"] = [
            x for x in items if x["name"] != "temp_to_psal"
        ]

        df = make_flag_frame(temp_qc_global_range=[4, 4])
        ds = load_nrtqc_step3_concat_dataset(nrtqc_config_001, df)
        ds.concat_flags()

        merged = ds.merged_data
        assert merged["temp_nrt_flag"].to_list() == [4, 4]
        assert merged["psal_nrt_flag"].to_list() == [1, 1]
        assert "psal_qc_temp_to_psal" not in merged.columns

    def test_unconfigured_columns_ignored(self, nrtqc_config_001):
        """Item columns not in the active item set do not contribute."""
        items = nrtqc_config_001.data["qc_item_set"]["items"]
        nrtqc_config_001.data["qc_item_set"]["items"] = [
            x for x in items if x["name"] not in ("spike", "temp_to_psal")
        ]

        df = make_flag_frame(temp_qc_spike=[4, 4])
        ds = load_nrtqc_step3_concat_dataset(nrtqc_config_001, df)
        ds.concat_flags()

        assert ds.merged_data["temp_nrt_flag"].to_list() == [1, 1]


class TestConcatErrorsAndOutput:
    """Error handling and output writing."""

    def test_concat_without_data_raises(self, nrtqc_config_001):
        ds = load_nrtqc_step3_concat_dataset(nrtqc_config_001, None)
        with pytest.raises(ValueError, match="qc_data"):
            ds.concat_flags()

    def test_write_without_concat_raises(self, nrtqc_config_001):
        ds = load_nrtqc_step3_concat_dataset(nrtqc_config_001, None)
        with pytest.raises(ValueError, match="merged_data"):
            ds.write_merged_data()

    def test_write(self, nrtqc_config_001):
        df = make_flag_frame(qc_impossible_date=[1, 1])
        ds = load_nrtqc_step3_concat_dataset(nrtqc_config_001, df)
        ds.concat_flags()
        ds.write_merged_data()

        assert os.path.exists(ds.output_file_name)
        os.remove(ds.output_file_name)
