"""QC flag dtype independence across the prepare steps.

Input sources disagree on how existing QC flags are stored: the R-generated
Parquet files carry integers, ``ctddump`` writes single-character strings
(``"1"``, ``"4"``, ``""`` for missing), and CSV/TSV inputs may arrive as
floats. The configuration is equally free to list flag values as ``[4]`` or
``["4"]``.

These tests run the profile-selection (step 3) and row-selection (step 4)
steps over the same fixture presented in each of those spellings and assert
the results are identical, so switching input source cannot silently change
which rows are selected or labelled.

Note on what is compared: the paired-row selection of ``LocateDataSetA``
picks, for each positive observation, the negative observation closest in
pressure, and resolves ties arbitrarily. Two runs over identical input can
therefore differ in which of several equally close rows they keep. Only the
flag-determined parts (the profiles, the positive rows, the row counts) are
compared frame-for-frame; the select-all variant, which has no pairing step,
is compared in full.
"""

import polars as pl
import pytest

from aiqclib.common.loader.dataset_loader import load_step3_select_dataset
from aiqclib.prepare.step4_select_rows.dataset_a import LocateDataSetA
from aiqclib.prepare.step4_select_rows.dataset_all import LocateDataSetAll

from tests.conftest import TARGETS


#: The existing QC flag columns of the test fixture.
FLAG_COLUMNS: tuple[str, ...] = tuple(f"{tgt}_qc" for tgt in TARGETS)

#: Spellings an input source may use for the same QC flags.
FLAG_DTYPES = pytest.mark.parametrize(
    "dtype",
    [pl.Utf8, pl.Float64, pl.Int64, pl.UInt8],
    ids=["string", "float", "int64", "uint8"],
)


def _recast_flags(input_data: pl.DataFrame, dtype: pl.DataType) -> pl.DataFrame:
    """Return the fixture with its QC flag columns cast to ``dtype``."""
    return input_data.with_columns([pl.col(name).cast(dtype) for name in FLAG_COLUMNS])


def _sorted(df: pl.DataFrame) -> pl.DataFrame:
    """Sort by every column so joins cannot make row order significant."""
    return df.sort(by=df.columns, nulls_last=True)


def _select_profiles(config, input_data: pl.DataFrame) -> pl.DataFrame:
    """Run step 3 and return the labelled profiles."""
    ds = load_step3_select_dataset(config, input_data=input_data)
    ds.label_profiles()
    return ds.selected_profiles


def _locate_paired(config, input_data: pl.DataFrame) -> dict:
    """Run step 3 + step 4 (paired variant) and return the rows per target."""
    ds = LocateDataSetA(
        config,
        input_data=input_data,
        selected_profiles=_select_profiles(config, input_data),
    )
    ds.process_targets()
    return ds.selected_rows


def _locate_all(config, input_data: pl.DataFrame) -> dict:
    """Run step 3 + step 4 (select-all variant) and return the rows per target."""
    ds = LocateDataSetAll(
        config,
        input_data=input_data,
        selected_profiles=_select_profiles(config, input_data),
    )
    ds.process_targets()
    return ds.selected_rows


class TestSelectProfiles:
    """Step 3 labels the same profiles whatever dtype the flags have."""

    @FLAG_DTYPES
    def test_profiles_match_the_integer_fixture(
        self, dtype, dataset_config_001, dataset_input_001
    ):
        """String, float and other integer flag columns select the same profiles."""
        reference = _select_profiles(dataset_config_001, dataset_input_001.input_data)
        recast = _select_profiles(
            dataset_config_001, _recast_flags(dataset_input_001.input_data, dtype)
        )
        assert _sorted(recast).equals(_sorted(reference))

    def test_reference_is_not_empty(self, dataset_config_001, dataset_input_001):
        """Guard against the comparison passing because nothing was selected."""
        reference = _select_profiles(dataset_config_001, dataset_input_001.input_data)
        assert reference.height > 0
        assert reference.filter(pl.col("label") == 1).height > 0


class TestSelectRowsAll:
    """Step 4 (select-all) is compared in full: it has no pairing step."""

    @FLAG_DTYPES
    def test_rows_match_the_integer_fixture(
        self, dtype, dataset_config_005, dataset_input_005
    ):
        """Every selected row, including its derived label, is unchanged."""
        reference = _locate_all(dataset_config_005, dataset_input_005.input_data)
        recast = _locate_all(
            dataset_config_005, _recast_flags(dataset_input_005.input_data, dtype)
        )
        for tgt in TARGETS:
            assert _sorted(recast[tgt]).equals(_sorted(reference[tgt]))

    def test_labels_are_flag_derived(self, dataset_config_005, dataset_input_005):
        """Guard: the label column must actually vary with the flags."""
        reference = _locate_all(dataset_config_005, dataset_input_005.input_data)
        labels = reference[TARGETS[0]]["label"].unique().to_list()
        assert set(labels) >= {0, 1}

    @FLAG_DTYPES
    def test_flag_column_is_always_int64(
        self, dtype, dataset_config_005, dataset_input_005
    ):
        """The emitted 'flag' column is Int64 regardless of the input dtype.

        Datasets from different sources are concatenated downstream, so the
        column needs one dtype whatever each source used.
        """
        rows = _locate_all(
            dataset_config_005, _recast_flags(dataset_input_005.input_data, dtype)
        )
        for tgt in TARGETS:
            assert rows[tgt].schema["flag"] == pl.Int64


class TestSelectRowsPaired:
    """Step 4 (paired): the flag-determined parts are unchanged."""

    @FLAG_DTYPES
    def test_positive_rows_match_the_integer_fixture(
        self, dtype, dataset_config_001, dataset_input_001
    ):
        """Positive rows come straight from the flag filter, so they must match."""
        reference = _locate_paired(dataset_config_001, dataset_input_001.input_data)
        recast = _locate_paired(
            dataset_config_001, _recast_flags(dataset_input_001.input_data, dtype)
        )
        for tgt in TARGETS:
            positives = pl.col("label") == 1
            assert _sorted(recast[tgt].filter(positives)).equals(
                _sorted(reference[tgt].filter(positives))
            )

    @FLAG_DTYPES
    def test_negative_row_counts_match(
        self, dtype, dataset_config_001, dataset_input_001
    ):
        """Which negative row wins a pressure tie is arbitrary; how many is not."""
        reference = _locate_paired(dataset_config_001, dataset_input_001.input_data)
        recast = _locate_paired(
            dataset_config_001, _recast_flags(dataset_input_001.input_data, dtype)
        )
        for tgt in TARGETS:
            negatives = pl.col("label") == 0
            assert (
                recast[tgt].filter(negatives).height
                == reference[tgt].filter(negatives).height
            )


class TestConfiguredFlagValues:
    """Flag values quoted in the configuration behave like the integer form."""

    def test_quoted_values_match_integer_values(
        self, dataset_config_005, dataset_input_005
    ):
        """['4'] selects exactly what [4] does, on a string flag column."""
        string_input = _recast_flags(dataset_input_005.input_data, pl.Utf8)
        reference = _locate_all(dataset_config_005, string_input)

        for target_value in dataset_config_005.get_target_dict().values():
            for key in ("pos_flag_values", "neg_flag_values"):
                target_value[key] = [str(value) for value in target_value[key]]

        quoted = _locate_all(dataset_config_005, string_input)

        for tgt in TARGETS:
            assert _sorted(quoted[tgt]).equals(_sorted(reference[tgt]))

    def test_non_numeric_value_is_rejected(self, dataset_config_005, dataset_input_005):
        """A flag value that is not a number is a configuration error, not a silent miss."""
        for target_value in dataset_config_005.get_target_dict().values():
            target_value["pos_flag_values"] = ["bad"]

        with pytest.raises(ValueError, match="whole number"):
            _locate_all(dataset_config_005, dataset_input_005.input_data)
