"""
This module provides the BasicValues class for extracting target
value observations from Polars DataFrames.

It extends FeatureBase and is designed for specific data processing needs,
such as those encountered with Copernicus CTD data.
"""

from typing import Optional, Dict

import polars as pl

from aiqclib.common.base.feature_base import FeatureBase


class BasicValues(FeatureBase):
    """
    A feature-extraction class for retrieving target values
    from Copernicus CTD data, extending :class:`FeatureBase`.
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
        Initialize an instance of BasicValues.

        :param target_name: The key identifying which target's rows to extract
                            features for from :attr:`selected_rows`. Defaults to :obj:`None`.
        :type target_name: Optional[str]
        :param feature_info: A dictionary containing feature-related parameters,
                             including a "stats" sub-dict with min/max info
                             and a "flank_up" integer specifying how many
                             upstream observations to retrieve. Defaults to :obj:`None`.
        :type feature_info: Optional[Dict]
        :param selected_profiles: A Polars DataFrame with selected profiles, typically
                                  used for further merges or lookups. Defaults to :obj:`None`.
        :type selected_profiles: Optional[pl.DataFrame]
        :param filtered_input: A potentially filtered Polars DataFrame containing
                               full observed variables. Defaults to :obj:`None`.
        :type filtered_input: Optional[pl.DataFrame]
        :param selected_rows: A dictionary mapping target names to their respective
                              DataFrames of relevant rows. Defaults to :obj:`None`.
        :type selected_rows: Optional[Dict[str, pl.DataFrame]]
        :param summary_stats: A Polars DataFrame of summary statistics
                              (unused in this subclass). Defaults to :obj:`None`.
        :type summary_stats: Optional[pl.DataFrame]
        """
        super().__init__(
            target_name=target_name,
            feature_info=feature_info,
            selected_profiles=selected_profiles,
            filtered_input=filtered_input,
            selected_rows=selected_rows,
            summary_stats=summary_stats,
        )
        self._expanded_observations: Optional[pl.DataFrame] = None
        self._feature_wide: Optional[pl.DataFrame] = None

    def extract_features(self) -> None:
        """
        Initiate the multi-step process of creating the feature set in :attr:`features`.

        Steps:

          1. :meth:`_init_features` - Prepare a base DataFrame with essential columns
             (row_id, platform_code, profile_no, observation_no).
          2. For each column specified in ``feature_info["col_names"]``,
             call :meth:`_add_features` to join the pivoted data onto our feature table.
          3. :meth:`_clean_features` - Drop columns no longer needed.
        """
        self._init_features()
        for col_name in self.feature_info["col_names"]:
            self._add_features(col_name)
        self._clean_features()

    def _init_features(self) -> None:
        """
        Initialize :attr:`features` by selecting core identifying columns
        from the DataFrame specified by :attr:`selected_rows[target_name]`.
        """
        self.features = self.selected_rows[self.target_name].select(
            ["row_id", "platform_code", "profile_no", "observation_no"]
        )

    def _add_features(self, col_name: str) -> None:
        """
        Join the specified column from :attr:`filtered_input` onto :attr:`features`
        using common identifier columns.

        :param col_name: The name of the column from :attr:`filtered_input` to add
                         as a new feature.
        :type col_name: str
        """
        self.features = self.features.join(
            (
                self.filtered_input.select(
                    pl.col("platform_code"),
                    pl.col("profile_no"),
                    pl.col("observation_no"),
                    pl.col(col_name),
                )
            ),
            on=["platform_code", "profile_no", "observation_no"],
            maintain_order="left",
        )

    def _clean_features(self) -> None:
        """
        Drop intermediate columns from :attr:`features` that are no longer
        needed after feature extraction and joining.
        """
        self.features = self.features.drop(
            ["platform_code", "profile_no", "observation_no"]
        )

    def scale_first(self) -> None:
        """
        Apply a pre-feature-extraction scaling step on :attr:`filtered_input`.

        This method performs min-max scaling derived from
        :attr:`feature_info["stats"]` for each relevant column,
        modifying :attr:`filtered_input` in place.
        """
        if self.feature_info["stats_set"]["type"] == "min_max":
            for col_name, v in self.feature_info["stats"].items():
                self.filtered_input = self.filtered_input.with_columns(
                    ((pl.col(col_name) - v["min"]) / (v["max"] - v["min"])).alias(
                        col_name
                    )
                )

    def scale_second(self) -> None:
        """
        Apply a post-feature-extraction scaling step if needed.

        This method is currently unimplemented but retains its placeholder
        for potential future additions of scaling or normalization
        after features have been pivoted and expanded.
        """
        pass  # pragma: no cover
