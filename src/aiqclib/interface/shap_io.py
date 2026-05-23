"""High-level interface for importing SHAP score files.

This module exposes :func:`read_shap_scores`, which loads the SHAP score files
produced by ``aiqclib`` during the testing and classification phases so they can
be used for SHAP visualization and evaluation (mean-importance bar charts,
summary plots, dependence plots, and so on).
"""

from typing import Any, Dict, Optional

import polars as pl

from aiqclib.common.utils.shap_io import read_shap_scores as _read_shap_scores


def read_shap_scores(
    file_name: str,
    file_type: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
    strip_suffix: bool = True,
) -> pl.DataFrame:
    """Import a SHAP score file produced by ``aiqclib``.

    ``aiqclib`` writes per-instance SHAP values with three metadata columns
    (``label``, ``predicted_label``, ``score``) followed by one ``<feature>_shap``
    column per feature. This function reads such a file into a Polars DataFrame
    and, by default, strips the ``_shap`` suffix so each feature column is named
    by its feature — convenient for downstream SHAP plots.

    :param file_name: Path to the SHAP score file.
    :type file_name: str
    :param file_type: Explicit file format (``"parquet"``, ``"tsv"``,
                      ``"tsv.gz"``, ``"csv"``, ``"csv.gz"``). Inferred from the
                      file extension when ``None``.
    :type file_type: Optional[str]
    :param options: Extra keyword arguments forwarded to the underlying Polars
                    reader.
    :type options: Optional[Dict[str, Any]]
    :param strip_suffix: Whether to strip the ``_shap`` suffix from the SHAP
                         columns. Defaults to ``True``.
    :type strip_suffix: bool
    :raises FileNotFoundError: If ``file_name`` does not exist.
    :raises ValueError: If the file type is unsupported, or if stripping the
                        suffix would produce duplicate column names.
    :return: A Polars DataFrame of SHAP scores.
    :rtype: polars.DataFrame
    """
    return _read_shap_scores(
        file_name,
        file_type=file_type,
        options=options,
        strip_suffix=strip_suffix,
    )
