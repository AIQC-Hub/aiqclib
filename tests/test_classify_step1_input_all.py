"""Unit tests for the ``InputDataSetAll`` class.

InputDataSetAll is the classify-side analogue of InputDataSetA: it reads
input data from a file path with optional file_type/read_file_options
configuration. Tests verify:
- Identity (step_name) and file path resolution from config
- Reading with explicit ``parquet`` file_type
- Reading with file_type inferred from extension
- Reading with empty/missing read_file_options
- FileNotFoundError on missing input
- ``n_rows`` option correctly limits returned rows

Refactored from a single ``unittest.TestCase`` class. The local
``_get_input_data`` helper is kept as a small module-level function that
takes the config, file_type, options and test_data_file path; replaces
the original's reliance on ``self.test_data_file``.
"""

import polars as pl
import pytest

from aiqclib.classify.step1_read_input.dataset_all import InputDataSetAll


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _read_input(config, test_data_file, *, file_type=None, read_file_options=None):
    """Construct InputDataSetAll, set the input file + options, read, return data.

    Pure helper — no test state. Used by the read-time tests that vary
    file_type and read_file_options in the config.
    """
    ds = InputDataSetAll(config)
    ds.input_file_name = str(test_data_file)

    if file_type is not None:
        ds.config.data["step_param_set"]["steps"]["input"]["file_type"] = file_type
    if read_file_options is not None:
        ds.config.data["step_param_set"]["steps"]["input"]["read_file_options"] = (
            read_file_options
        )

    ds.read_input_data()
    return ds.input_data


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestInputDataSetAll:
    """Tests for InputDataSetAll's identity, file path resolution, and reading."""

    def test_step_name(self, classify_config_001):
        """step_name == 'input'."""
        ds = InputDataSetAll(classify_config_001)
        assert ds.step_name == "input"

    def test_input_file_name(self, classify_config_001):
        """input_file_name comes from config.path_info; no data-dependence."""
        ds = InputDataSetAll(classify_config_001)
        assert (
            str(ds.input_file_name)
            == "/path/to/input_1/input_folder_1/nrt_cora_bo_test.parquet"
        )

    def test_read_input_data_with_explicit_type(self, classify_config_001, test_data_file):
        """Reading succeeds when 'parquet' is explicitly set as file_type."""
        df = _read_input(
            classify_config_001, test_data_file,
            file_type="parquet", read_file_options={},
        )
        assert isinstance(df, pl.DataFrame)
        assert df.shape[0] == 2456
        assert df.shape[1] == 30

    def test_read_input_data_infer_type(self, classify_config_001, test_data_file):
        """Reading succeeds when file_type is inferred from the file extension."""
        df = _read_input(
            classify_config_001, test_data_file,
            file_type=None, read_file_options={},
        )
        assert isinstance(df, pl.DataFrame)
        assert df.shape[0] == 2456
        assert df.shape[1] == 30

    def test_read_input_data_missing_options(self, classify_config_001, test_data_file):
        """Reading succeeds when read_file_options is None."""
        df = _read_input(
            classify_config_001, test_data_file,
            file_type="parquet", read_file_options=None,
        )
        assert isinstance(df, pl.DataFrame)
        assert df.shape[0] == 2456
        assert df.shape[1] == 30

    def test_read_input_data_file_not_found(self, classify_config_001, test_data_file):
        """Reading from a non-existent path raises FileNotFoundError."""
        ds = InputDataSetAll(classify_config_001)
        ds.input_file_name = str(test_data_file) + "_not_found"

        with pytest.raises(FileNotFoundError):
            ds.read_input_data()

    def test_read_input_data_with_extra_options(self, classify_config_001, test_data_file):
        """Polars read_file_options (e.g. ``n_rows=100``) are honoured."""
        df = _read_input(
            classify_config_001, test_data_file,
            file_type="parquet", read_file_options={"n_rows": 1000},
        )
        assert isinstance(df, pl.DataFrame)
        # n_rows=100 limits the result regardless of the underlying file size.
        assert df.shape[0] == 288
        assert df.shape[1] == 30