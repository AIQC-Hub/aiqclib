"""
Module for testing the `read_input_file` utility function.

This module contains a series of unit tests for the `read_input_file` function,
covering various scenarios such as reading different file types (parquet, csv, tsv,
and gzipped versions), inferring file types, handling unsupported file types,
managing non-existent files, and passing additional reader options.
"""

import unittest
from pathlib import Path

import polars as pl

from aiqclib.common.utils.file import read_input_file


class TestReadInputFile(unittest.TestCase):
    """
    A suite of tests verifying various file reading scenarios
    (multiple file types, type inference, non-existent files, etc.)
    using 'read_input_file'.
    """

    def setUp(self):
        """
        Runs once before any tests. Sets the directory
        where test data files are located.
        """
        self.test_data_dir = Path(__file__).resolve().parent / "data" / "input"

    def test_read_input_file_explicit_type(self):
        """
        Tests reading various file types when 'file_type' is explicitly provided.
        Verifies correct row count and DataFrame type.
        """
        test_cases = [
            ("nrt_cora_bo_test.parquet", 132342, "parquet"),
            ("nrt_cora_bo_test_2023_row1.csv", 1, "csv"),
            ("nrt_cora_bo_test_2023_row1.tsv", 1, "tsv"),
            ("nrt_cora_bo_test_2023_row1.csv.gz", 1, "csv.gz"),
            ("nrt_cora_bo_test_2023_row1.tsv.gz", 1, "tsv.gz"),
        ]
        for file_name, expected_rows, file_type in test_cases:
            with self.subTest(file_name=file_name, file_type=file_type):
                file_path = self.test_data_dir / file_name
                df = read_input_file(file_path, file_type=file_type, options={})
                self.assertIsInstance(df, pl.DataFrame)
                self.assertEqual(df.shape[0], expected_rows)
                self.assertEqual(df.shape[1], 30)

    def test_read_input_file_infer_type(self):
        """
        Tests reading various file types when 'file_type' is inferred from the file extension.
        Verifies correct row count and DataFrame type.
        """
        test_cases = [
            ("nrt_cora_bo_test.parquet", 132342),
            ("nrt_cora_bo_test_2023_row1.csv", 1),
            ("nrt_cora_bo_test_2023_row1.tsv", 1),
            ("nrt_cora_bo_test_2023_row1.csv.gz", 1),
            ("nrt_cora_bo_test_2023_row1.tsv.gz", 1),
        ]
        for file_name, expected_rows in test_cases:
            with self.subTest(file_name=file_name):
                file_path = self.test_data_dir / file_name
                df = read_input_file(file_path, file_type=None, options={})
                self.assertIsInstance(df, pl.DataFrame)
                self.assertEqual(df.shape[0], expected_rows)
                self.assertEqual(df.shape[1], 30)

    def test_unsupported_file_type(self):
        """
        Verifies that passing an explicitly unsupported 'file_type' raises a ValueError.
        """
        file_path = self.test_data_dir / "nrt_cora_bo_test.parquet"
        with self.assertRaises(ValueError) as context:
            _ = read_input_file(file_path, file_type="foo", options={})
        self.assertIn("Unsupported file_type 'foo'", str(context.exception))

    def test_non_existent_file(self):
        """
        Verifies that attempting to read a non-existent file raises a FileNotFoundError.
        """
        with self.assertRaises(FileNotFoundError):
            _ = read_input_file(
                Path("non_existent_file.csv"), file_type="csv", options={}
            )

    def test_pass_additional_options(self):
        """
        Tests passing additional reader-specific options, like 'has_header=False',
        to ensure they are correctly applied.
        """
        file_name = "nrt_cora_bo_test_2023_row1.csv.gz"
        file_path = self.test_data_dir / file_name
        df = read_input_file(
            file_path, file_type="csv.gz", options={"has_header": False}
        )
        self.assertIsInstance(df, pl.DataFrame)

    def test_empty_options(self):
        """
        Tests that passing 'None' as options argument does not cause issues
        and the file can still be read.
        """
        file_name = "nrt_cora_bo_test_2023_row1.csv.gz"
        file_path = self.test_data_dir / file_name
        df = read_input_file(file_path, file_type="csv.gz", options=None)
        self.assertIsInstance(df, pl.DataFrame)

    def test_file_type_inference_unsupported_extension(self):
        """
        Verifies that attempting to read a file with an unhandled or unsupported
        extension (e.g., '.txt') without explicit type raises a ValueError during inference.
        """
        file_name = "empty_text_file.txt"
        file_path = self.test_data_dir / file_name
        with self.assertRaises(ValueError):
            _ = read_input_file(file_path)
