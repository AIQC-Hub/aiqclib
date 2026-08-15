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


def exclude_from_final_flag(config, *item_names) -> None:
    """Mark the named items ``include_in_final_flag: false`` in place."""
    by_name = {x["name"]: x for x in config.data["qc_item_set"]["items"]}
    for name in item_names:
        by_name[name]["include_in_final_flag"] = False


class TestFinalFlagItemSelection:
    """``include_in_final_flag: false`` drops an item from the aggregation.

    The distinction under test throughout: the item still runs and its
    column is still written, so only the final flag changes.
    """

    def test_excluded_item_does_not_raise_the_final_flag(self, nrtqc_config_001):
        """A failing excluded item leaves the final flag good."""
        exclude_from_final_flag(nrtqc_config_001, "spike")

        df = make_flag_frame(temp_qc_spike=[4, 4], temp_qc_global_range=[1, 1])
        ds = load_nrtqc_step3_concat_dataset(nrtqc_config_001, df)
        ds.concat_flags()

        merged = ds.merged_data
        assert merged["temp_nrt_flag"].to_list() == [1, 1]
        # The column survives: the failure is recorded, it just does not vote.
        assert merged["temp_qc_spike"].to_list() == [4, 4]

    def test_included_items_still_decide(self, nrtqc_config_001):
        """Excluding one item leaves the others deciding the final flag."""
        exclude_from_final_flag(nrtqc_config_001, "spike")

        df = make_flag_frame(temp_qc_spike=[4, 4], temp_qc_global_range=[1, 3])
        ds = load_nrtqc_step3_concat_dataset(nrtqc_config_001, df)
        ds.concat_flags()

        assert ds.merged_data["temp_nrt_flag"].to_list() == [1, 3]

    def test_exclusion_is_per_item_not_per_column(self, nrtqc_config_001):
        """Excluding an item drops it for every variable it flags."""
        exclude_from_final_flag(nrtqc_config_001, "global_range")

        df = make_flag_frame(
            temp_qc_global_range=[4, 1],
            psal_qc_global_range=[4, 1],
            psal_qc_spike=[1, 3],
        )
        ds = load_nrtqc_step3_concat_dataset(nrtqc_config_001, df)
        ds.concat_flags()

        merged = ds.merged_data
        assert merged["temp_nrt_flag"].to_list() == [1, 1]
        assert merged["psal_nrt_flag"].to_list() == [1, 3]

    def test_profile_level_item_can_be_excluded(self, nrtqc_config_001):
        """A variable-independent item is excluded for all variables."""
        exclude_from_final_flag(nrtqc_config_001, "impossible_date")

        df = make_flag_frame(qc_impossible_date=[4, 4])
        ds = load_nrtqc_step3_concat_dataset(nrtqc_config_001, df)
        ds.concat_flags()

        merged = ds.merged_data
        assert merged["temp_nrt_flag"].to_list() == [1, 1]
        assert merged["psal_nrt_flag"].to_list() == [1, 1]
        assert merged["qc_impossible_date"].to_list() == [4, 4]

    def test_excluding_every_item_yields_good_flags(self, nrtqc_config_001):
        """With nothing included the final flags fall back to good."""
        exclude_from_final_flag(nrtqc_config_001, *nrtqc_config_001.get_qc_item_names())

        df = make_flag_frame(qc_impossible_date=[4, 4], temp_qc_spike=[4, 3])
        ds = load_nrtqc_step3_concat_dataset(nrtqc_config_001, df)
        ds.concat_flags()

        assert ds.merged_data["temp_nrt_flag"].to_list() == [1, 1]
        assert ds.merged_data["psal_nrt_flag"].to_list() == [1, 1]

    def test_default_matches_previous_behaviour(self, nrtqc_config_001):
        """Without the key set, aggregation is unchanged."""
        df = make_flag_frame(temp_qc_spike=[4, 1], qc_impossible_date=[1, 3])
        ds = load_nrtqc_step3_concat_dataset(nrtqc_config_001, df)
        ds.concat_flags()

        assert ds.merged_data["temp_nrt_flag"].to_list() == [4, 3]


class TestDeferredPropagationExclusion:
    """Excluding the deferred temp_to_psal item computes but does not raise."""

    def test_column_written_but_flag_untouched(self, nrtqc_config_001):
        """psal_qc_temp_to_psal is still produced; psal_nrt_flag is not raised."""
        exclude_from_final_flag(nrtqc_config_001, "temp_to_psal")

        df = make_flag_frame(temp_qc_global_range=[4, 3])
        ds = load_nrtqc_step3_concat_dataset(nrtqc_config_001, df)
        ds.concat_flags()

        merged = ds.merged_data
        assert merged["temp_nrt_flag"].to_list() == [4, 3]
        # The propagation is computed and traceable ...
        assert merged["psal_qc_temp_to_psal"].to_list() == [4, 3]
        # ... but does not decide psal's verdict.
        assert merged["psal_nrt_flag"].to_list() == [1, 1]

    def test_psal_own_items_still_decide(self, nrtqc_config_001):
        """Excluding propagation leaves psal's own items in charge."""
        exclude_from_final_flag(nrtqc_config_001, "temp_to_psal")

        df = make_flag_frame(temp_qc_global_range=[4, 4], psal_qc_spike=[1, 3])
        ds = load_nrtqc_step3_concat_dataset(nrtqc_config_001, df)
        ds.concat_flags()

        assert ds.merged_data["psal_nrt_flag"].to_list() == [1, 3]

    def test_excluding_the_source_item_changes_what_propagates(self, nrtqc_config_001):
        """Propagation reads the aggregated flag, so excluding its inputs matters.

        temp_to_psal consumes ``temp_nrt_flag`` rather than the raw item
        columns, so dropping an item from temp's aggregation also removes it
        from what reaches psal.
        """
        exclude_from_final_flag(nrtqc_config_001, "spike")

        df = make_flag_frame(temp_qc_spike=[4, 4])
        ds = load_nrtqc_step3_concat_dataset(nrtqc_config_001, df)
        ds.concat_flags()

        merged = ds.merged_data
        assert merged["temp_nrt_flag"].to_list() == [1, 1]
        assert merged["psal_qc_temp_to_psal"].to_list() == [1, 1]
        assert merged["psal_nrt_flag"].to_list() == [1, 1]


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
