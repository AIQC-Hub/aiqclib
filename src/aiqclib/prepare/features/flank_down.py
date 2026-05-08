"""
This module provides the FlankDown class for extracting "flanking" (neighboring) observations
around target rows within Copernicus CTD datasets. It specializes in downstream
observation expansion and feature pivoting.
"""

from typing import Optional, Dict

import polars as pl

from aiqclib.common.base.feature_base import FeatureBase


class FlankDown(FeatureBase):
    """
    A feature-extraction class for retrieving target values and their "flanking" values
    from Copernicus CTD data, extending :class:`FeatureBase`.

    The term "flanking values" refers to the concept of capturing neighboring observations
    around a specified index (e.g., observation_no) by shifting backward a specified amount.
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
        Initialize an instance of FlankDown.

        :param target_name: The key identifying which target's rows to extract
                            features for from :attr:`selected_rows`.
        :type target_name: Optional[str]
        :param feature_info: A dictionary containing feature-related parameters,
                             including a "stats" sub-dict with min/max info
                             and a "flank_down" integer specifying how many
                             downstream observations to retrieve.
        :type feature_info: Optional[Dict]
        :param selected_profiles: A Polars DataFrame with selected profiles, typically
                                  used for further merges or lookups.
        :type selected_profiles: Optional[pl.DataFrame]
        :param filtered_input: A potentially filtered Polars DataFrame containing
                               full observed variables.
        :type filtered_input: Optional[pl.DataFrame]
        :param selected_rows: A dictionary mapping target names to their respective
                              DataFrames of relevant rows.
        :type selected_rows: Optional[Dict[str, pl.DataFrame]]
        :param summary_stats: A Polars DataFrame of summary statistics
                              (unused in this subclass).
        :type summary_stats: Optional[pl.DataFrame]
        :return: None
        :rtype: None
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
          1. :meth:`_init_features` - Prepare a base DataFrame with essential columns.
          2. :meth:`_expand_observations` - Expand observations based on "flank_down".
          3. For each column in ``feature_info["col_names"]``:
             - :meth:`_pivot_features` to pivot the data.
             - :meth:`_add_features` to join the pivoted data onto the feature table.
          4. :meth:`_clean_features` - Drop metadata columns.

        :return: None
        :rtype: None
        """
        self._init_features()
        self._expand_observations()
        for col_name in self.feature_info["col_names"]:
            self._pivot_features(col_name)
            self._add_features()
        self._clean_features()

    def _init_features(self) -> None:
        """
        Initialize :attr:`features` by selecting core columns
        from :attr:`selected_rows[target_name]`.

        :return: None
        :rtype: None
        """
        self.features = self.selected_rows[self.target_name].select(
            ["row_id", "platform_code", "profile_no"]
        )

    def _expand_observations(self) -> None:
        """
        Generate a DataFrame with additional rows for each "flank" step.

        This expands each row in :attr:`selected_rows[target_name]` by
        cross joining with a sequence from 1 to ``feature_info["flank_down"]``,
        then adjusts ``observation_no`` to shift forwards for each flank step.

        :return: None
        :rtype: None
        """
        summary_df = self.filtered_input.group_by(
            pl.col("platform_code"),
            pl.col("profile_no"),
        ).agg(pl.len().alias("profile_count"))

        self._expanded_observations = (
            self.selected_rows[self.target_name]
            .select(["row_id", "platform_code", "profile_no", "observation_no"])
            .join(
                pl.DataFrame(
                    {
                        "flank_seq": list(
                            range(1, self.feature_info.get("flank_down") + 1)
                        )
                    }
                ),
                how="cross",
            )
            .join(summary_df, how="inner", on=["platform_code", "profile_no"])
            .with_columns(
                (pl.col("observation_no") + pl.col("flank_seq")).alias("observation_no")
            )
            .with_columns(
                pl.when(pl.col("observation_no") > pl.col("profile_count"))
                .then(pl.col("profile_count"))
                .otherwise(pl.col("observation_no"))
                .alias("observation_no")
            )
        )

    def _pivot_features(self, col_name: str) -> None:
        """
        Pivot the expanded observations to create columns for each flank step
        of the specified data column.

        :param col_name: The original data column to be pivoted (e.g., "temp").
        :type col_name: str
        :return: None
        :rtype: None
        """
        self._feature_wide = (
            self._expanded_observations.join(
                self.filtered_input.select(
                    pl.col("platform_code"),
                    pl.col("profile_no"),
                    pl.col("observation_no"),
                    pl.col(col_name).alias("value"),
                ),
                on=["platform_code", "profile_no", "observation_no"],
                maintain_order="left",
            )
            .with_columns(
                pl.concat_str(
                    [
                        pl.lit(f"{col_name}_down"),
                        pl.col("flank_seq").cast(pl.Utf8),
                    ],
                    separator="_",
                ).alias("col_name")
            )
            .drop(["observation_no", "flank_seq"])
            .pivot(
                "col_name",
                index=["row_id", "platform_code", "profile_no"],
                values="value",
            )
        )

    def _add_features(self) -> None:
        """
        Join the pivoted columns from :attr:`_feature_wide` onto :attr:`features`.

        :return: None
        :rtype: None
        """
        self.features = self.features.join(
            self._feature_wide,
            on=["row_id", "platform_code", "profile_no"],
            maintain_order="left",
        )

    def _clean_features(self) -> None:
        """
        Drop columns that are no longer needed in the final feature set.

        :return: None
        :rtype: None
        """
        self.features = self.features.drop(["platform_code", "profile_no"])

    def scale_first(self) -> None:
        """
        Apply a pre-feature-extraction scaling step on :attr:`filtered_input`
        using min-max scaling derived from :attr:`feature_info["stats"]`.

        :return: None
        :rtype: None
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
        Currently unimplemented.

        :return: None
        :rtype: None
        """
        pass  # pragma: no cover
