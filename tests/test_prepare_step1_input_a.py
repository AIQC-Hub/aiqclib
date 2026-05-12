"""Unit tests for the ``InputDataSetA`` class.

Exercises ``InputDataSetA``'s ability to read Parquet input, resolve input
file paths from config, apply column renames, and filter rows by year.

Refactored from three ``unittest.TestCase`` classes with separate ``setUp``
blocks and a triplicated ``_get_input_data`` helper into three plain test
classes that share the helper at module level and pull config + data-file
paths from conftest fixtures.
"""

import polars as pl
import pytest

from aiqclib.prepare.step1_read_input.dataset_a import InputDataSetA


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _read(config, test_data_file, file_type=None, read_file_options=None):
    """Build an InputDataSetA, optionally override step params, read, return df.

    Previously duplicated verbatim across three TestCase classes. Kept as a
    plain function at module scope — it's pure logic with no test state.
    """
    ds = InputDataSetA(config)
    ds.input_file_name = str(test_data_file)

    if file_type is not None:
        ds.config.data["step_param_set"]["steps"]["input"]["file_type"] = file_type

    if read_file_options is not None:
        ds.config.data["step_param_set"]["steps"]["input"][
            "read_file_options"
        ] = read_file_options

    ds.read_input_data()
    return ds.input_data


# ---------------------------------------------------------------------------
# Basic reading behaviour
# ---------------------------------------------------------------------------

class TestInputDataSetA:
    """Reading Parquet input, resolving file names, and config-overrideable options."""

    def test_step_name(self, dataset_config_001):
        """InputDataSetA reports step_name == 'input'."""
        ds = InputDataSetA(dataset_config_001)
        assert ds.step_name == "input"

    def test_input_file_name(self, dataset_config_001):
        """The default input_file_name is derived from config.path_info."""
        ds = InputDataSetA(dataset_config_001)
        assert (
            str(ds.input_file_name)
            == "/path/to/input_1/input_folder_1/nrt_cora_bo_test.parquet"
        )

    def test_read_input_data_with_explicit_type(self, dataset_config_001, test_data_file):
        """An explicit ``file_type='parquet'`` produces a DataFrame of the expected shape."""
        df = _read(dataset_config_001, test_data_file, file_type="parquet", read_file_options={})
        assert isinstance(df, pl.DataFrame)
        assert df.shape[0] == 1524
        assert df.shape[1] == 30

    def test_read_input_data_infer_type(self, dataset_config_001, test_data_file):
        """File type is inferred from extension when not set explicitly."""
        df = _read(dataset_config_001, test_data_file, file_type=None, read_file_options={})
        assert isinstance(df, pl.DataFrame)
        assert df.shape[0] == 1524
        assert df.shape[1] == 30

    def test_read_input_data_missing_options(self, dataset_config_001, test_data_file):
        """Missing read_file_options is equivalent to an empty dict."""
        df = _read(dataset_config_001, test_data_file, file_type="parquet", read_file_options=None)
        assert isinstance(df, pl.DataFrame)
        assert df.shape[0] == 1524
        assert df.shape[1] == 30

    def test_read_input_data_file_not_found(self, dataset_config_001, test_data_file):
        """A non-existent path raises FileNotFoundError."""
        ds = InputDataSetA(dataset_config_001)
        ds.input_file_name = str(test_data_file) + "_not_found"
        with pytest.raises(FileNotFoundError):
            ds.read_input_data()

    def test_read_input_data_with_extra_options(self, dataset_config_001, test_data_file):
        """``read_file_options={'n_rows': 100}`` caps the read at 100 rows.

        100 is below the reduced fixture size so this assertion should be
        stable across fixture changes — but bump the cap if the fixture ever
        drops below 100 rows.
        """
        df = _read(
            dataset_config_001, test_data_file,
            file_type="parquet", read_file_options={"n_rows": 100},
        )
        assert isinstance(df, pl.DataFrame)
        assert df.shape[0] == 100
        assert df.shape[1] == 30


# ---------------------------------------------------------------------------
# Column-rename behaviour (uses dataset config 002, which sets rename_dict)
# ---------------------------------------------------------------------------

class TestInputDataSetARename:
    """``rename_dict`` in config swaps column names; removing it leaves originals."""

    def test_rename(self, dataset_config_002, test_data_file):
        """``filename`` is renamed to ``filename_new`` per the YAML's rename_dict."""
        df = _read(dataset_config_002, test_data_file, file_type="parquet", read_file_options={})
        assert "filename" not in df.columns
        assert "filename_new" in df.columns

    def test_rename_with_incorrect_param(self, dataset_config_002, test_data_file):
        """Deleting rename_dict from the step params leaves the original column."""
        del dataset_config_002.get_step_params("input")["rename_dict"]
        df = _read(dataset_config_002, test_data_file, file_type="parquet", read_file_options={})
        assert "filename" in df.columns


# ---------------------------------------------------------------------------
# Year-filtering behaviour (also config 002, exercises filter_rows/keep/remove)
# ---------------------------------------------------------------------------

def _uniq_years(df: pl.DataFrame) -> list[int]:
    """Distinct years from the ``profile_timestamp`` column, as a list."""
    return (
        df.select(pl.col("profile_timestamp").dt.year().unique())
          .to_series()
          .to_list()
    )


class TestInputDataSetAFilter:
    """``filter_rows`` plus ``remove_years``/``keep_years`` shape the year set."""

    def test_remove_years_without_filter_rows_flag(self, dataset_config_002, test_data_file):
        """When filter_rows=False, no year filtering occurs."""
        dataset_config_002.get_step_params("input")["sub_steps"]["filter_rows"] = False
        df = _read(dataset_config_002, test_data_file, file_type="parquet", read_file_options={})
        assert _uniq_years(df) == [2021, 2023]

    def test_remove_years_with_empty_array(self, dataset_config_002, test_data_file):
        """Empty remove_years + empty keep_years means no filtering."""
        params = dataset_config_002.get_step_params("input")
        params["sub_steps"]["filter_rows"] = True
        params["filter_method_dict"]["remove_years"] = []
        params["filter_method_dict"]["keep_years"] = []

        df = _read(dataset_config_002, test_data_file, file_type="parquet", read_file_options={})
        assert _uniq_years(df) == [2021, 2023]

    def test_remove_years(self, dataset_config_002, test_data_file):
        """``remove_years=[2022, 2023]`` excludes those two years."""
        params = dataset_config_002.get_step_params("input")
        params["sub_steps"]["filter_rows"] = True
        params["filter_method_dict"]["remove_years"] = [2022, 2023]
        params["filter_method_dict"]["keep_years"] = []

        df = _read(dataset_config_002, test_data_file, file_type="parquet", read_file_options={})
        assert _uniq_years(df) == [2021]

    def test_keep_years(self, dataset_config_002, test_data_file):
        """``keep_years=[2022, 2023]`` retains only those years."""
        params = dataset_config_002.get_step_params("input")
        params["sub_steps"]["filter_rows"] = True
        params["filter_method_dict"]["remove_years"] = []
        params["filter_method_dict"]["keep_years"] = [2022, 2023]

        df = _read(dataset_config_002, test_data_file, file_type="parquet", read_file_options={})
        assert _uniq_years(df) == [2023]
