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
``ensure_output_file`` adds the refusal to replace an existing file.

Refactored from a ``unittest.TestCase`` class with ``self.subTest`` loops
inside two test methods. The subTest loops become ``@pytest.mark.parametrize``,
sharing a module-level case list between the explicit-type and inferred-type
tests.
"""

import os

import pytest
import polars as pl

from aiqclib.common.utils.file import (
    ensure_output_directory,
    ensure_output_file,
    expand_path,
    read_input_file,
)


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

    def test_tilde_path_is_expanded(self, tmp_path, monkeypatch):
        """An input path written as '~/file.csv' reads from the home directory."""
        monkeypatch.setenv("HOME", str(tmp_path))
        (tmp_path / "data.csv").write_text("a,b\n1,2\n")

        df = read_input_file("~/data.csv")

        assert df.shape == (1, 2)


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

    def test_tilde_is_expanded_before_the_directory_is_checked(self):
        """'~' means the home directory, so the message names the real path."""
        with pytest.raises(IOError) as excinfo:
            ensure_output_directory("~/aiqclib_no_such_dir/out.yaml")

        message = str(excinfo.value)
        assert os.path.expanduser("~/aiqclib_no_such_dir") in message
        assert "~" not in message

    def test_tilde_directory_is_created_under_home(self, tmp_path, monkeypatch):
        """create_dirs=True creates the directory under home, not under the CWD."""
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.chdir(tmp_path / "..")

        result = ensure_output_directory("~/made/here/out.yaml", create_dirs=True)

        assert result == str(tmp_path / "made" / "here")
        assert (tmp_path / "made" / "here").is_dir()
        # The bug this guards against: a literal '~' folder in the working directory.
        assert not (tmp_path / ".." / "~").exists()


class TestEnsureOutputFile:
    """Refusing to replace a file that is already there."""

    def test_new_file_is_allowed(self, tmp_path):
        """Nothing to protect, so the path is returned unchanged."""
        target = tmp_path / "out.yaml"
        assert ensure_output_file(str(target)) == str(target)

    def test_existing_file_raises_by_default(self, tmp_path):
        """An existing file is kept, and the message names the option."""
        target = tmp_path / "out.yaml"
        target.write_text("customized")
        with pytest.raises(FileExistsError, match="overwrite=True"):
            ensure_output_file(str(target))
        assert target.read_text() == "customized"

    def test_overwrite_allows_replacing(self, tmp_path):
        """With the flag set the caller may replace the file."""
        target = tmp_path / "out.yaml"
        target.write_text("customized")
        assert ensure_output_file(str(target), overwrite=True) == str(target)

    def test_directory_in_place_of_a_file(self, tmp_path):
        """A directory where the file should be is reported for what it is."""
        target = tmp_path / "out.yaml"
        target.mkdir()
        with pytest.raises(IsADirectoryError, match="is a directory"):
            ensure_output_file(str(target), overwrite=True)

    def test_directory_check_still_applies(self, tmp_path):
        """The missing-directory refusal is not lost by the new check."""
        with pytest.raises(IOError, match="create_dirs=True"):
            ensure_output_file(str(tmp_path / "missing" / "out.yaml"))

    def test_create_dirs_and_overwrite_combine(self, tmp_path):
        """Both options can apply to the same call."""
        target = tmp_path / "a" / "b" / "out.yaml"
        ensure_output_file(str(target), create_dirs=True)
        target.write_text("first")
        ensure_output_file(str(target), create_dirs=True, overwrite=True)
        assert target.parent.is_dir()

    def test_returns_the_expanded_path(self, tmp_path, monkeypatch):
        """The caller writes to the path that was checked, not the raw one."""
        monkeypatch.setenv("HOME", str(tmp_path))

        result = ensure_output_file("~/out.yaml")

        assert result == str(tmp_path / "out.yaml")

    def test_existing_file_is_detected_through_a_tilde(self, tmp_path, monkeypatch):
        """The overwrite guard sees the real file, not an unexpanded name."""
        monkeypatch.setenv("HOME", str(tmp_path))
        (tmp_path / "out.yaml").write_text("customized")

        with pytest.raises(FileExistsError, match="overwrite=True"):
            ensure_output_file("~/out.yaml")


class TestExpandPath:
    """Turning a user-written '~' into the directory they meant."""

    def test_leading_tilde_becomes_the_home_directory(self, tmp_path, monkeypatch):
        """The common case: '~/x' is the home directory, not a folder named '~'."""
        monkeypatch.setenv("HOME", str(tmp_path))
        assert expand_path("~/aiqc_project/data") == str(tmp_path / "aiqc_project/data")

    def test_expanded_path_is_absolute(self, tmp_path, monkeypatch):
        """An unexpanded '~' path is relative, which is the underlying bug."""
        monkeypatch.setenv("HOME", str(tmp_path))
        assert os.path.isabs(expand_path("~/aiqc_project"))

    @pytest.mark.parametrize(
        "path",
        [
            "/absolute/path/data.parquet",
            "relative/path/data.parquet",
            "data.parquet",
            "",
        ],
    )
    def test_paths_without_a_tilde_are_untouched(self, path):
        """Expansion only applies to '~'; everything else passes through."""
        assert expand_path(path) == path

    def test_tilde_not_at_the_start_is_untouched(self):
        """A '~' inside a filename is an ordinary character."""
        assert (
            expand_path("/data/backup~1/file.parquet") == "/data/backup~1/file.parquet"
        )
