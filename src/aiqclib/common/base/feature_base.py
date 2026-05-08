"""
Standardized Feature Extraction and Scaling Module.

This module defines the `FeatureBase` abstract base class (ABC), which provides a
standardized framework for feature engineering tasks using the Polars library.
It ensures that subclasses implement a consistent pipeline for feature
extraction and multi-stage scaling.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional

import polars as pl


class FeatureBase(ABC):
    """
    Abstract base class for extracting and scaling features.

    Child classes must implement all abstract methods to define specific
    logic for feature generation and normalization. This class serves as a
    container for the data and metadata required during the transformation
    lifecycle.

    :ivar target_name: Name of the target variable.
    :ivar feature_info: Metadata or configuration for features.
    :ivar selected_profiles: Polars DataFrame of pre-selected profiles.
    :ivar filtered_input: Polars DataFrame of pre-filtered input data.
    :ivar selected_rows: Mapping of identifiers to specific Polars DataFrames.
    :ivar summary_stats: Polars DataFrame containing summary statistics.
    :ivar features: Polars DataFrame containing the processed features.
    """

    def __init__(
        self,
        target_name: Optional[str] = None,
        feature_info: Optional[Dict] = None,
        selected_profiles: Optional[pl.DataFrame] = None,
        filtered_input: Optional[pl.DataFrame] = None,
        selected_rows: Optional[Dict[str, pl.DataFrame]] = None,
        summary_stats: Optional[pl.DataFrame] = None,
    ) -> None:
        """
        Initialize the feature-extraction base class with optional data and metadata.

        :param target_name: Name of the target variable to use when extracting features.
        :type target_name: Optional[str]
        :param feature_info: A dictionary containing metadata or configuration about features.
        :type feature_info: Optional[Dict]
        :param selected_profiles: A Polars DataFrame containing pre-selected profiles.
        :type selected_profiles: Optional[pl.DataFrame]
        :param filtered_input: A Polars DataFrame that may already include filters.
        :type filtered_input: Optional[pl.DataFrame]
        :param selected_rows: A dictionary mapping identifiers to Polars DataFrames.
        :type selected_rows: Optional[Dict[str, pl.DataFrame]]
        :param summary_stats: A Polars DataFrame of summary statistics for transformations.
        :type summary_stats: Optional[pl.DataFrame]
        :return: None
        :rtype: None
        """
        self.target_name: Optional[str] = target_name
        self.feature_info: Optional[Dict] = feature_info
        self.selected_profiles: Optional[pl.DataFrame] = selected_profiles
        self.filtered_input: Optional[pl.DataFrame] = filtered_input
        self.selected_rows: Optional[Dict[str, pl.DataFrame]] = selected_rows
        self.summary_stats: Optional[pl.DataFrame] = summary_stats
        self.features: Optional[pl.DataFrame] = None

    @abstractmethod
    def extract_features(self) -> None:
        """
        Extract features from the provided data sources.

        This method must be implemented by subclasses to generate raw features
        from inputs like `filtered_input` or `selected_rows`. The resulting
        DataFrame should be assigned to `self.features`.

        :return: None
        :rtype: None
        """
        pass  # pragma: no cover

    @abstractmethod
    def scale_first(self) -> None:
        """
        Apply the first pass of scaling or normalization to the extracted features.

        Typically used for initial transformations such as standard scaling or
        handling outliers. This method should update the `self.features` attribute.

        :return: None
        :rtype: None
        """
        pass  # pragma: no cover

    @abstractmethod
    def scale_second(self) -> None:
        """
        Apply a secondary scaling or refinement step to the features.

        Used for additional adjustments or domain-specific normalizations
        required after the first scaling pass. This method should update
        the `self.features` attribute.

        :return: None
        :rtype: None
        """
        pass  # pragma: no cover
