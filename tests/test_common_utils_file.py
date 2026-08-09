"""Unit tests for the file utilities: ``read_input_file`` and
``ensure_output_directory``.

read_input_file is the polymorphic reader the pipeline uses to load
parquet, CSV, TSV, and gzipped CSV/TSV files. Tests verify:
- All five supported file types load correctly with an explicit ``file_type``
- The same file types also load correctly with inferred ``file_type``
- An unsupported explicit ``file_type`` raises ValueError with a descriptive
  message
- A non-existent file raises FileNotFoundError
- Additional polars options (e.g. ``has_header``) flow through
- ``options=None`` is equivalent to ``options={}``
- A file with an unsupported extension and no explicit type raises ValueError

``ensure_output_directory`` guards the destination of a file about to be
written: it refuses a missing directory by default and creates it on request.

Refactored from a ``unittest.TestCase`` class with ``self.subTest`` loops
inside two test methods. The subTest loops become ``@pytest.mark.parametrize``,
sharing a module-level case list between the explicit-type and inferred-type
tests.
"""

import pytest
import polars as pl

from aiqclib.common.utils.file import ensure_output_directory, read_input_file


# ---------------------------------------------------------------------------
# Shared test-case data
# ---------------------------------------------------------------------------

# Each tuple is (file_name, expected_rows, file_type) — used by both the
# explicit-type and inferred-type parametrized tests. The inferred-type
# test passes ``file_type=None``, ignoring the third element of each tuple.

# the test-data reduction. If the user has reduced this specific file (and
# not just the train/test split outputs), update to the actual current
# row count. Reference: the row count for input_data shape assertions in
# test_prepare_step1_input_a.py uses the same number — both should match.
_PARQUET_ROW_COUNT = 3267

# CSV/TSV files have ``_row1`` in their filename indicating a single row.
_SINGLE_ROW = 1

# All test inputs have 30 columns regardless of format.
_COLUMN_COUNT = 30

_READ_INPUT_CASES = [
    ("nrt_cora_bo_test.parquet", _PARQUET_ROW_COUNT, "parquet"),
    ("nrt_cora_bo_test_2023_row1.csv", _SINGLE_ROW, "csv"),
    ("nrt_cora_bo_test_2023_row1.tsv", _SINGLE_ROW, "tsv"),
    ("nrt_cora_bo_test_2023_row1.csv.gz", _SINGLE_ROW, "csv.gz"),
    ("nrt_cora_bo_test_2023_row1.tsv.gz", _SINGLE_ROW, "tsv.gz"),
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestReadInputFile:
    """Tests for read_input_file across file types and option configurations."""

    @pytest.mark.parametrize(
        "file_name, expected_rows, file_type",
        _READ_INPUT_CASES,
        ids=[case[0] for case in _READ_INPUT_CASES],
    )
    def test_read_input_file_explicit_type(
        self,
        file_name,
        expected_rows,
        file_type,
        input_dir,
    ):
        """Each supported file type reads correctly with explicit file_type."""
        df = read_input_file(input_dir / file_name, file_type=file_type, options={})
        assert isinstance(df, pl.DataFrame)
        assert df.shape[0] == expected_rows
        assert df.shape[1] == _COLUMN_COUNT

    @pytest.mark.parametrize(
        "file_name, expected_rows, file_type",
        _READ_INPUT_CASES,
        ids=[case[0] for case in _READ_INPUT_CASES],
    )
    def test_read_input_file_infer_type(
        self,
        file_name,
        expected_rows,
        file_type,
        input_dir,
    ):
        """Each supported file type also reads with file_type inferred from extension.

        The third tuple element (``file_type``) is unused here — kept in the
        parametrize signature so the test cases stay sharable with the
        explicit-type test. ``_`` would obscure the column meaning.
        """
        df = read_input_file(input_dir / file_name, file_type=None, options={})
        assert isinstance(df, pl.DataFrame)
        assert df.shape[0] == expected_rows
        assert df.shape[1] == _COLUMN_COUNT

    def test_unsupported_file_type(self, input_dir):
        """Explicit unsupported file_type raises ValueError with a descriptive message."""
        with pytest.raises(ValueError, match="Unsupported file_type 'foo'"):
            _ = read_input_file(
                input_dir / "nrt_cora_bo_test.parquet",
                file_type="foo",
                options={},
            )

    def test_non_existent_file(self):
        """A non-existent file path raises FileNotFoundError."""
        from pathlib import Path

        with pytest.raises(FileNotFoundError):
            _ = read_input_file(
                Path("non_existent_file.csv"),
                file_type="csv",
                options={},
            )

    def test_pass_additional_options(self, input_dir):
        """Polars-specific options (e.g. has_header=False) flow through correctly."""
        df = read_input_file(
            input_dir / "nrt_cora_bo_test_2023_row1.csv.gz",
            file_type="csv.gz",
            options={"has_header": False},
        )
        assert isinstance(df, pl.DataFrame)

    def test_empty_options(self, input_dir):
        """``options=None`` is treated the same as ``options={}`` — file still reads."""
        df = read_input_file(
            input_dir / "nrt_cora_bo_test_2023_row1.csv.gz",
            file_type="csv.gz",
            options=None,
        )
        assert isinstance(df, pl.DataFrame)

    def test_file_type_inference_unsupported_extension(self, input_dir):
        """A file with an unsupported extension and no explicit type raises ValueError.

        The inference path can't match e.g. ``.txt`` to any of the known
        readers, so it surfaces the same kind of error as an explicit
        unsupported file_type.
        """
        with pytest.raises(ValueError):
            _ = read_input_file(input_dir / "empty_text_file.txt")


class TestEnsureOutputDirectory:
    """Guarding the destination directory of a file about to be written."""

    def test_existing_directory_passes(self, tmp_path):
        """An existing directory is returned unchanged and nothing is created."""
        result = ensure_output_directory(str(tmp_path / "out.yaml"))
        assert result == str(tmp_path)

    def test_no_directory_part(self, tmp_path):
        """A bare filename has no directory to check."""
        assert ensure_output_directory("out.yaml") == ""

    def test_missing_directory_raises_by_default(self, tmp_path):
        """The default refuses to create anything, and says how to opt in."""
        target = tmp_path / "missing" / "out.yaml"
        with pytest.raises(IOError, match="create_dirs=True"):
            ensure_output_directory(str(target))
        assert not target.parent.exists()

    def test_create_dirs_creates_nested_directories(self, tmp_path):
        """Every missing parent is created, not just the last component."""
        target = tmp_path / "a" / "b" / "c" / "out.yaml"
        result = ensure_output_directory(str(target), create_dirs=True)
        assert result == str(target.parent)
        assert target.parent.is_dir()

    def test_create_dirs_is_idempotent(self, tmp_path):
        """Calling twice does not raise on the second, now-existing, directory."""
        target = tmp_path / "a" / "out.yaml"
        ensure_output_directory(str(target), create_dirs=True)
        ensure_output_directory(str(target), create_dirs=True)
        assert target.parent.is_dir()

    @pytest.mark.parametrize("create_dirs", [False, True])
    def test_path_blocked_by_a_file(self, tmp_path, create_dirs):
        """A regular file where the directory should be is reported clearly."""
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory")
        with pytest.raises(IOError, match="is not a directory"):
            ensure_output_directory(str(blocker / "out.yaml"), create_dirs=create_dirs)

    def test_tilde_path_reports_that_it_is_not_expanded(self):
        """'~' looks right but is literal, so the message calls it out."""
        with pytest.raises(IOError, match="not expanded automatically"):
            ensure_output_directory("~/aiqclib_no_such_dir/out.yaml")
