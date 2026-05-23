"""
Tests for the SHAP score importer
(:func:`aiqclib.common.utils.shap_io.read_shap_scores` and its interface
wrapper). Synthetic files only, so these run without the downloaded
``tests/data`` fixtures.
"""

import polars as pl
import pytest

import aiqclib as aq
from aiqclib.common.utils.shap_io import read_shap_scores


def _write_shap_parquet(path):
    """A small SHAP file mirroring the real output: label/predicted_label/score
    plus several ``<feature>_shap`` columns."""
    df = pl.DataFrame(
        {
            "label": [0, 1],
            "predicted_label": [1, 0],
            "score": [0.94, 0.10],
            "longitude_shap": [0.0, 0.2],
            "temp_mean_shap": [0.0, -0.5],
            "psal_sd_shap": [2.78, 0.0],
            "pres_down_5_shap": [0.0, 0.1],
        }
    )
    df.write_parquet(path)
    return df


def test_exported_from_package():
    assert "read_shap_scores" in aq.__all__
    assert callable(aq.read_shap_scores)


def test_strips_shap_suffix_and_keeps_metadata(tmp_path):
    path = tmp_path / "classify_shap_values_temp.parquet"
    _write_shap_parquet(path)

    out = read_shap_scores(str(path))

    # Metadata columns are untouched.
    for col in ("label", "predicted_label", "score"):
        assert col in out.columns
    # No SHAP suffix remains; feature columns are renamed.
    assert not any(c.endswith("_shap") for c in out.columns)
    assert {"longitude", "temp_mean", "psal_sd", "pres_down_5"} <= set(out.columns)
    # Values are preserved.
    assert out["psal_sd"].to_list() == [2.78, 0.0]
    assert out["score"].to_list() == [0.94, 0.10]


def test_strip_suffix_false_keeps_original_names(tmp_path):
    path = tmp_path / "shap.parquet"
    _write_shap_parquet(path)

    out = read_shap_scores(str(path), strip_suffix=False)
    assert "temp_mean_shap" in out.columns
    assert "temp_mean" not in out.columns


def test_interface_matches_util(tmp_path):
    path = tmp_path / "shap.parquet"
    _write_shap_parquet(path)
    assert aq.read_shap_scores(str(path)).columns == read_shap_scores(str(path)).columns


def test_csv_input_is_supported(tmp_path):
    path = tmp_path / "shap.csv"
    path.write_text(
        "label,predicted_label,score,temp_mean_shap\n0,1,0.5,0.3\n", encoding="utf-8"
    )
    out = read_shap_scores(str(path))
    assert "temp_mean" in out.columns
    assert out["temp_mean"].to_list() == [0.3]


def test_duplicate_after_strip_raises(tmp_path):
    # A bare 'temp' column alongside 'temp_shap' would collide once stripped.
    path = tmp_path / "shap.parquet"
    pl.DataFrame({"temp": [1.0], "temp_shap": [0.2]}).write_parquet(path)

    with pytest.raises(ValueError) as excinfo:
        read_shap_scores(str(path))
    assert "temp" in str(excinfo.value)
    # Still importable without stripping.
    assert "temp_shap" in read_shap_scores(str(path), strip_suffix=False).columns


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_shap_scores(str(tmp_path / "nope.parquet"))


def test_no_shap_columns_returns_unchanged(tmp_path):
    path = tmp_path / "plain.parquet"
    pl.DataFrame({"label": [0], "score": [0.5]}).write_parquet(path)
    out = read_shap_scores(str(path))
    assert out.columns == ["label", "score"]
