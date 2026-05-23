"""
Utilities for importing the SHAP score files produced by ``aiqclib``.

During the testing (training) and classification phases, ``aiqclib`` can write
per-instance SHAP values to a Parquet file. Each such file has three metadata
columns — ``label``, ``predicted_label`` and ``score`` — followed by one column
per feature, each suffixed with ``_shap`` (e.g. ``temp_mean_shap``,
``longitude_shap``).

This module reads such a file back into a Polars DataFrame and, by default,
strips the ``_shap`` suffix from the feature columns so the result can be fed
straight into common SHAP visualizations (mean-importance bar charts, summary
plots, dependence plots, etc.).
"""

from typing import Any, Dict, Optional

import polars as pl

from aiqclib.common.utils.file import read_input_file

#: Suffix used by ``aiqclib`` for columns that hold SHAP values.
SHAP_COLUMN_SUFFIX = "_shap"


def read_shap_scores(
    file_name: str,
    file_type: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
    strip_suffix: bool = True,
    suffix: str = SHAP_COLUMN_SUFFIX,
) -> pl.DataFrame:
    """
    Import a SHAP score file written by ``aiqclib``.

    The file is read via :func:`aiqclib.common.utils.file.read_input_file`, so
    Parquet, TSV and CSV (optionally gzipped) inputs are all supported and the
    format is inferred from the extension when ``file_type`` is not given.

    By default, the ``_shap`` suffix is removed from every SHAP-value column, so
    each feature column is named by the feature itself (e.g. ``temp_mean_shap``
    becomes ``temp_mean``). The metadata columns (``label``, ``predicted_label``,
    ``score``) do not carry the suffix and are returned unchanged.

    :param file_name: Path to the SHAP score file.
    :type file_name: str
    :param file_type: Explicit file format (``"parquet"``, ``"tsv"``,
                      ``"tsv.gz"``, ``"csv"``, ``"csv.gz"``). Inferred from the
                      extension when ``None``.
    :type file_type: Optional[str]
    :param options: Extra keyword arguments forwarded to the underlying Polars
                    reader.
    :type options: Optional[Dict[str, Any]]
    :param strip_suffix: Whether to strip the SHAP suffix from column names.
                         Defaults to ``True``.
    :type strip_suffix: bool
    :param suffix: The suffix identifying SHAP-value columns. Defaults to
                   :data:`SHAP_COLUMN_SUFFIX`.
    :type suffix: str
    :raises FileNotFoundError: If ``file_name`` does not exist.
    :raises ValueError: If the file type is unsupported, or if stripping the
                        suffix would produce duplicate column names.
    :returns: A Polars DataFrame of SHAP scores, with feature columns renamed
              unless ``strip_suffix`` is ``False``.
    :rtype: pl.DataFrame

    Example:
      >>> import aiqclib as aq
      >>> shap = aq.read_shap_scores("classify_shap_values_temp.parquet")
      >>> # Feature columns (everything that carried the _shap suffix):
      >>> features = [c for c in shap.columns
      ...             if c not in ("label", "predicted_label", "score")]
      >>> # Matrix of SHAP values for plotting (e.g. mean importance, summary,
      >>> # or dependence plots):
      >>> values = shap.select(features).to_numpy()
    """
    df = read_input_file(file_name, file_type=file_type, options=options)

    if not strip_suffix:
        return df

    rename_map = {
        column: column[: -len(suffix)]
        for column in df.columns
        if column.endswith(suffix) and len(column) > len(suffix)
    }
    if not rename_map:
        return df

    final_names = [rename_map.get(column, column) for column in df.columns]
    duplicates = sorted({name for name in final_names if final_names.count(name) > 1})
    if duplicates:
        raise ValueError(
            f"Stripping '{suffix}' from column names would create duplicate "
            f"column(s): {', '.join(duplicates)}. Call with strip_suffix=False "
            f"to keep the original names."
        )

    return df.rename(rename_map)
