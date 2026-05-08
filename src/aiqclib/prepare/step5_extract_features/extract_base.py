"""
This module provides the ExtractFeatureBase abstract base class, designed for
orchestrating feature extraction workflows using Polars. It facilitates
the processing of dataset targets, dynamic loading of feature extraction
logic, and the persistence of generated features to disk.
"""

import os
from typing import Dict, Optional

import polars as pl

from aiqclib.common.base.config_base import ConfigBase
from aiqclib.common.base.dataset_base import DataSetBase
from aiqclib.common.loader.feature_loader import load_feature_class


class ExtractFeatureBase(DataSetBase):
    """
    Abstract base class for extracting features from dataset rows.

    This class provides the core framework for managing data, applying feature
    extraction logic, and saving the results. It inherits from :class:`DataSetBase`
    to ensure configuration consistency and utilizes a feature loader to dynamically
    compose feature extraction steps. The extracted features, once generated,
    can be written to Parquet files.
    """

    def __init__(
        self,
        config: ConfigBase,
        input_data: Optional[pl.DataFrame] = None,
        selected_profiles: Optional[pl.DataFrame] = None,
        selected_rows: Optional[Dict[str, pl.DataFrame]] = None,
        summary_stats: Optional[pl.DataFrame] = None,
    ) -> None:
        """
        Initialize the feature extraction base class.

        :param config: The configuration object, containing paths and target definitions.
        :type config: ConfigBase
        :param input_data: A Polars DataFrame providing the full dataset from which
                           features are extracted.
        :type input_data: Optional[pl.DataFrame]
        :param selected_profiles: A Polars DataFrame containing profiles that have
                                 been selected for processing.
        :type selected_profiles: Optional[pl.DataFrame]
        :param selected_rows: A dictionary mapping target names to Polars DataFrames
                            of rows relevant for those targets.
        :type selected_rows: Optional[Dict[str, pl.DataFrame]]
        :param summary_stats: A Polars DataFrame containing summary statistics that
                              might guide feature scaling.
        :type summary_stats: Optional[pl.DataFrame]
        """
        super().__init__(step_name="extract", config=config)

        #: The default pattern to use when writing feature files for each target.
        self.default_file_name: str = "extracted_features_{target_name}.parquet"

        #: A dictionary mapping target names to corresponding output Parquet file paths.
        self.output_file_names: Dict[str, str] = self.config.get_target_file_names(
            step_name="extract", default_file_name=self.default_file_name
        )

        self.input_data: Optional[pl.DataFrame] = input_data
        self.selected_profiles: Optional[pl.DataFrame] = selected_profiles

        # Filter input data if both input_data and selected_profiles are present
        if input_data is not None and selected_profiles is not None:
            self._filter_input()
        else:
            self.filtered_input: Optional[pl.DataFrame] = None

        #: A dict of Polars DataFrames, one per target, indicating rows to be used.
        self.selected_rows: Optional[Dict[str, pl.DataFrame]] = selected_rows
        #: A Polars DataFrame presenting summary stats for optional use in scaling features.
        self.summary_stats: Optional[pl.DataFrame] = summary_stats
        #: A dictionary specifying feature extraction parameters from the config.
        self.feature_info: Dict = self.config.data["feature_param_set"]["params"]
        #: A dictionary mapping target names to DataFrames of extracted features.
        self.target_features: Dict[str, pl.DataFrame] = {}

        #: Column names used for intermediate processing (e.g., to maintain
        #: matching references between positive and negative rows). These columns
        #: will be dropped from the final feature set.
        self.drop_col_names: list[str] = []

    def _filter_input(self) -> None:
        """
        Filter the input data by joining with the selected profiles.

        This method ensures that the resulting :attr:`filtered_input` only
        contains rows also present in :attr:`selected_profiles`. The data
        is joined on columns ``platform_code`` and ``profile_no``.
        """
        self.filtered_input = self.input_data.join(
            self.selected_profiles.select(
                pl.col("platform_code"),
                pl.col("profile_no"),
            ).unique(),
            on=["platform_code", "profile_no"],
        )

    def process_targets(self) -> None:
        """
        Generate features for all targets found in the configuration.

        Iterates over each target name returned by
        :meth:`~aiqclib.common.base.config_base.ConfigBase.get_target_names`
        and calls :meth:`extract_target_features` on them.
        """
        for target_name in self.config.get_target_names():
            self.extract_target_features(target_name)

    def extract_target_features(self, target_name: str) -> None:
        """
        Build the features for a specified target.

        This method retrieves the relevant rows for the given target, extracts
        features using the configured feature information, and then joins them
        with essential metadata columns. Finally, it drops any specified temporary
        columns.

        :param target_name: The key identifying which target to process.
        :type target_name: str
        """
        self.target_features[target_name] = (
            self.selected_rows[target_name]
            .select(
                [
                    "row_id",
                    "label",
                    "profile_id",
                    "pair_id",
                    "platform_code",
                    "profile_no",
                    "observation_no",
                ]
            )
            .join(
                pl.concat(
                    [
                        self.extract_features(target_name, fi)
                        for fi in self.feature_info
                    ],
                    how="align_left",
                ),
                on=["row_id"],
                maintain_order="left",
            )
        )

        self.target_features[target_name] = self.target_features[target_name].drop(
            self.drop_col_names
        )

    def extract_features(self, target_name: str, feature_info: Dict) -> pl.DataFrame:
        """
        Use a feature loader to retrieve and run a feature extraction process.

        This method dynamically loads a feature extraction class based on the
        provided `feature_info`, passes the relevant data, and then executes
        the scaling and extraction steps defined within that class.

        :param target_name: The target for which features will be extracted.
        :type target_name: str
        :param feature_info: A dictionary of feature extraction parameters.
        :type feature_info: Dict
        :return: A DataFrame containing newly extracted or transformed features.
        :rtype: pl.DataFrame
        """
        ds = load_feature_class(
            target_name,
            feature_info,
            self.selected_profiles,
            self.filtered_input,
            self.selected_rows,
            self.summary_stats,
        )

        ds.scale_first()
        ds.extract_features()
        ds.scale_second()

        return ds.features

    def write_target_features(self) -> None:
        """
        Write the extracted features to their respective files.

        Iterates through the :attr:`target_features` dictionary and writes each
        Polars DataFrame to a Parquet file, creating necessary directories.

        :raises ValueError: If :attr:`target_features` is empty.
        """
        if not self.target_features:
            raise ValueError("Member variable 'target_features' must not be empty.")

        for target, df in self.target_features.items():
            output_path = self.output_file_names[target]
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df.write_parquet(output_path)
