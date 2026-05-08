"""
This module provides utility functions for reading various file formats into Polars DataFrames.

It supports common data formats like Parquet, TSV (tab-separated values), and CSV
(comma-separated values), including their gzipped versions, and allows for automatic
file type inference based on file extensions.
"""

import os
from typing import Dict, Any, Optional

import polars as pl


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
