"""
This module provides utilities for dynamically loading and instantiating
feature extraction classes from a predefined registry. It serves as a central
point for retrieving specific feature implementations based on configuration
details, facilitating a modular approach to feature engineering.
"""

from typing import Dict, Optional

import polars as pl

from aiqclib.common.base.feature_base import FeatureBase
from aiqclib.common.loader.feature_registry import FEATURE_REGISTRY


def load_feature_class(
    target_name: str,
    feature_info: Dict,
    selected_profiles: Optional[pl.DataFrame] = None,
    filtered_input: Optional[pl.DataFrame] = None,
    selected_rows: Optional[pl.DataFrame] = None,
    summary_stats: Optional[pl.DataFrame] = None,
) -> FeatureBase:
    """Instantiate a feature extraction class using the specified feature registry.

    This function retrieves the class name from the ``"feature"`` key within
    the provided ``feature_info`` dictionary. It then looks up the
    corresponding class in the global :data:`FEATURE_REGISTRY` and
    instantiates it with the supplied parameters.

    :param target_name: The target variable or dataset name for which features
                        will be extracted. This is typically a column name
                        or a unique identifier for the data subset.
    :type target_name: str
    :param feature_info: A dictionary describing the feature extraction procedure.
                         This dictionary must at least include the key ``"feature"``
                         whose value is the string name of the feature class
                         registered in :data:`FEATURE_REGISTRY`.
    :type feature_info: Dict
    :param selected_profiles: An optional Polars DataFrame containing specific
                              profiles (e.g., sample IDs, experiment runs)
                              relevant to the current feature extraction task.
                              Defaults to None.
    :type selected_profiles: Optional[pl.DataFrame]
    :param filtered_input: An optional Polars DataFrame containing data that
                           has already been winnowed to relevant observations
                           for advanced merging or lookups within the feature
                           extraction process. Defaults to None.
    :type filtered_input: Optional[pl.DataFrame]
    :param selected_rows: An optional Polars DataFrame containing the specific
                          rows or observations that are the focus for this
                          target's feature calculation. Defaults to None.
    :type selected_rows: Optional[pl.DataFrame]
    :param summary_stats: An optional Polars DataFrame containing pre-computed
                          summary statistics (e.g., mean, standard deviation)
                          for potential use in scaling, normalization, or
                          transformation steps during feature extraction.
                          Defaults to None.
    :type summary_stats: Optional[pl.DataFrame]
    :return: An instance of the requested feature extraction class, which
             must inherit from :class:`FeatureBase`.
    :rtype: FeatureBase
    :raises ValueError: If the ``"feature"`` key is missing from ``feature_info``
                        or if the specified feature class name is not found
                        in :data:`FEATURE_REGISTRY`.
    """
    class_name = feature_info.get("feature")
    if not class_name:  # Explicitly check for missing "feature" key
        raise ValueError("Missing 'feature' key in feature_info dictionary.")

    feature_class = FEATURE_REGISTRY.get(class_name)
    if not feature_class:
        raise ValueError(f"Unknown feature class specified: {class_name}")

    return feature_class(
        target_name,
        feature_info,
        selected_profiles,
        filtered_input,
        selected_rows,
        summary_stats,
    )
