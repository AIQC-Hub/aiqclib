"""
This module provides file utilities: reading various file formats into Polars
DataFrames, and preparing the destination of a file that is about to be written.

It supports common data formats like Parquet, TSV (tab-separated values), and CSV
(comma-separated values), including their gzipped versions, and allows for automatic
file type inference based on file extensions.
"""

import os
from typing import Dict, Any, Optional

import polars as pl


def expand_path(file_name: str) -> str:
    """
    Expand a leading ``~`` (or ``~user``) in a path.

    Paths reach the library from two directions: arguments passed to the public
    functions, and ``base_path`` values read out of a YAML configuration. A user
    writing either one reasonably expects ``~/aiqc_project`` to mean their home
    directory, but nothing expands it on their behalf — YAML is not a shell, and
    ``os.path.join`` treats ``~`` as an ordinary directory name. Left alone, a
    path like that resolves relative to the current working directory, so output
    lands in a literal ``~`` folder wherever the interpreter happened to start.

    Applying this at the points where paths enter the library keeps that
    expectation true without every call site having to remember it.

    :param file_name: A path that may begin with ``~``.
    :type file_name: str
    :returns: The path with a leading ``~`` replaced by the user's home
              directory, unchanged when there is nothing to expand.
    :rtype: str
    """
    return os.path.expanduser(file_name)


def ensure_output_directory(file_name: str, create_dirs: bool = False) -> str:
    """
    Check that the directory of a file about to be written exists.

    Writing into a directory that does not exist is refused by default, so a
    mistyped path fails immediately instead of scattering files across new
    directories. Pass ``create_dirs=True`` to create the missing directories
    instead; the raised message names the option, so a caller who meant to
    create them does not have to look it up.

    :param file_name: The path (including filename) of the file to be written.
    :type file_name: str
    :param create_dirs: If True, create the directory (and any missing parents)
                        rather than raising. Defaults to False.
    :type create_dirs: bool
    :raises IOError: If the directory does not exist and ``create_dirs`` is
                     False, or if the path exists but is not a directory.
    :returns: The directory part of ``file_name`` after ``~`` expansion, empty
              when it has none.
    :rtype: str
    """
    dir_path = os.path.dirname(expand_path(file_name))
    if dir_path == "" or os.path.isdir(dir_path):
        return dir_path

    if os.path.exists(dir_path):
        raise IOError(f"'{dir_path}' exists but is not a directory.")

    if not create_dirs:
        raise IOError(
            f"Directory '{dir_path}' does not exist. Create it first, or pass "
            f"create_dirs=True to create it automatically."
        )

    os.makedirs(dir_path, exist_ok=True)
    return dir_path


def ensure_output_file(
    file_name: str, create_dirs: bool = False, overwrite: bool = False
) -> str:
    """
    Prepare the destination of a file about to be written.

    Checks the directory (see :func:`ensure_output_directory`) and then refuses
    to replace an existing file unless ``overwrite`` is set. Refusing by default
    matters for files a user edits after they are generated: silently rewriting
    one throws away that work with nothing to undo it.

    :param file_name: The path (including filename) of the file to be written.
    :type file_name: str
    :param create_dirs: If True, create a missing output directory rather than
                        raising. Defaults to False.
    :type create_dirs: bool
    :param overwrite: If True, replace an existing file. Defaults to False.
    :type overwrite: bool
    :raises IOError: If the directory does not exist and ``create_dirs`` is False.
    :raises FileExistsError: If the file exists and ``overwrite`` is False.
    :raises IsADirectoryError: If the path names an existing directory.
    :returns: ``file_name`` with ``~`` expanded, so the caller writes to the
              same path that was checked here.
    :rtype: str
    """
    ensure_output_directory(file_name, create_dirs=create_dirs)
    file_name = expand_path(file_name)

    if os.path.isdir(file_name):
        raise IsADirectoryError(
            f"'{file_name}' is a directory, so it cannot be written as a file."
        )

    if os.path.exists(file_name) and not overwrite:
        raise FileExistsError(
            f"File '{file_name}' already exists. Pass overwrite=True to replace "
            f"it, or choose another path. Refusing by default so an edited file "
            f"is not lost."
        )

    return file_name


def read_input_file(
    input_file: str,
    file_type: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
) -> pl.DataFrame:
    """
    Read an input file into a Polars DataFrame, supporting formats such as
    Parquet, TSV (optionally gzipped), and CSV (optionally gzipped).

    :param input_file: The full path to the file to be read.
    :type input_file: str
    :param file_type: The file format. Must be one of:
                      - "parquet"
                      - "tsv"
                      - "tsv.gz"
                      - "csv"
                      - "csv.gz"

                      If set to None or an empty string, the file type is inferred from
                      the file extension. Defaults to None.
    :type file_type: Optional[str]
    :param options: A dictionary of additional keyword arguments to pass to
                    the Polars reading function (e.g., "has_header", "infer_schema_length").
                    Defaults to None.
    :type options: Optional[Dict[str, Any]]
    :raises FileNotFoundError: If the specified ``input_file`` does not exist.
    :raises ValueError: If the file type cannot be inferred or is not supported.
    :returns: A Polars DataFrame containing the contents of the file.
    :rtype: pl.DataFrame

    Example Usage:
      >>> import polars as pl
      >>> # Assuming 'data.parquet' and 'data.tsv.gz' exist for demonstration
      >>> # df = read_input_file("data.parquet")
      >>> # df2 = read_input_file("data.tsv.gz", file_type="tsv.gz", options={"has_header": True})
    """
    input_file = expand_path(input_file)
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"File '{input_file}' does not exist.")

    if options is None:
        options = {}

    # Infer file type based on file extension if not provided.
    if not file_type:
        filename = os.path.basename(input_file).lower()
        if filename.endswith(".parquet"):
            file_type = "parquet"
        elif filename.endswith(".tsv.gz"):
            file_type = "tsv.gz"
        elif filename.endswith(".tsv"):
            file_type = "tsv"
        elif filename.endswith(".csv.gz"):
            file_type = "csv.gz"
        elif filename.endswith(".csv"):
            file_type = "csv"
        else:
            raise ValueError(
                "Could not infer file type automatically. Please specify 'file_type' explicitly."
            )

    # Read the file using the appropriate Polars function.
    if file_type == "parquet":
        df = pl.read_parquet(input_file, **options)
    elif file_type in ("tsv", "tsv.gz"):
        df = pl.read_csv(input_file, separator="\t", **options)
    elif file_type in ("csv", "csv.gz"):
        df = pl.read_csv(input_file, **options)
    else:
        raise ValueError(
            f"Unsupported file_type '{file_type}'. Must be one of: "
            "'parquet', 'tsv', 'tsv.gz', 'csv', 'csv.gz'."
        )

    return df
