"""
This module defines the ExtractDataSetProfile class, the profile-level
feature extraction step. Profile-native features are extracted directly on
the one-row-per-profile frame; observation-level features are extracted on
the target's observation rows and aggregated per profile via the ``agg``
key of their feature parameters.
"""

from typing import Dict, Optional

import polars as pl

from aiqclib.common.base.config_base import ConfigBase
from aiqclib.common.constants import PROFILE_KEYS
from aiqclib.common.loader.feature_loader import load_feature_class
from aiqclib.common.loader.feature_registry import FEATURE_REGISTRY
from aiqclib.common.utils.aggregations import (
    build_aggregation_exprs,
    validate_aggregation_names,
)
from aiqclib.prepare.step5_extract_features.extract_base import ExtractFeatureBase


class ExtractDataSetProfile(ExtractFeatureBase):
    """
    A subclass of :class:`ExtractFeatureBase` that builds one feature row per
    profile for the profile-level pipeline.

    For each entry of the feature parameters, the feature class's ``level``
    decides the route:

    - ``"profile"``: the feature runs unchanged against the per-profile
      :attr:`selected_rows` and its columns are joined directly.
    - ``"observation"``: the entry must carry an ``agg`` list; the feature
      runs against the target's :attr:`observation_rows` and each feature
      column is aggregated per profile into ``{column}_{agg}`` columns. An
      observation-level feature without ``agg`` is rejected; this is what
      keeps un-aggregated observation-level NRT QC items out of
      profile-level training.
    """

    expected_class_name: str = "ExtractDataSetProfile"

    def __init__(
        self,
        config: ConfigBase,
        input_data: Optional[pl.DataFrame] = None,
        selected_profiles: Optional[pl.DataFrame] = None,
        selected_rows: Optional[Dict[str, pl.DataFrame]] = None,
        summary_stats: Optional[pl.DataFrame] = None,
        observation_rows: Optional[Dict[str, pl.DataFrame]] = None,
    ) -> None:
        """
        Initialize the profile-level feature extraction step.

        :param config: A dataset configuration object that manages paths,
                       target definitions, and parameters.
        :type config: :class:`~aiqclib.common.base.config_base.ConfigBase`
        :param input_data: An optional Polars DataFrame containing the full
                           pre-processed dataset.
        :type input_data: :class:`polars.DataFrame` or None
        :param selected_profiles: An optional Polars DataFrame of selected profiles.
        :type selected_profiles: :class:`polars.DataFrame` or None
        :param selected_rows: A mapping of target names to one-row-per-profile
                              frames produced by the profile-level locate step.
        :type selected_rows: Dict[str, :class:`polars.DataFrame`] or None
        :param summary_stats: An optional Polars DataFrame with summary
                              statistics for feature scaling.
        :type summary_stats: :class:`polars.DataFrame` or None
        :param observation_rows: A mapping of target names to the valid-flagged
                                 observation rows kept by the locate step, used
                                 to run and aggregate observation-level features.
        :type observation_rows: Dict[str, :class:`polars.DataFrame`] or None
        """
        super().__init__(
            config=config,
            input_data=input_data,
            selected_profiles=selected_profiles,
            selected_rows=selected_rows,
            summary_stats=summary_stats,
        )
        self.observation_rows = observation_rows

    def extract_target_features(self, target_name: str) -> None:
        """
        Build the profile-level features for a specified target.

        The per-profile spine (identity columns plus label) is joined with
        the direct profile-level features on ``row_id`` and with the
        aggregated observation-level features on the profile keys.

        :param target_name: The key identifying which target to process.
        :type target_name: str
        :raises ValueError: If two features produce the same output column
                            (e.g. ``profile_summary_stats`` with ``mean`` and
                            an aggregated feature with ``agg: [mean]`` both
                            yield ``{variable}_mean``).
        """
        spine = self.selected_rows[target_name].select(
            [
                "row_id",
                "label",
                "profile_id",
                "pair_id",
                "platform_code",
                "profile_no",
            ]
        )

        profile_frames = []
        aggregated_frames = []
        for feature_info in self.feature_info:
            if self._feature_level(feature_info) == "profile":
                profile_frames.append(self.extract_features(target_name, feature_info))
            else:
                aggregated_frames.append(
                    self._extract_aggregated_features(target_name, feature_info)
                )

        self._check_column_collisions(profile_frames, aggregated_frames)

        features = spine
        if profile_frames:
            features = features.join(
                pl.concat(profile_frames, how="align_left"),
                on=["row_id"],
                maintain_order="left",
            )
        for aggregated in aggregated_frames:
            features = features.join(aggregated, on=PROFILE_KEYS, maintain_order="left")

        self.target_features[target_name] = features.drop(self.drop_col_names)

    @staticmethod
    def _check_column_collisions(profile_frames: list, aggregated_frames: list) -> None:
        """
        Raise when two feature frames produce the same output column.

        Collisions arise most easily between per-profile summary statistics
        (``{variable}_{metric}``) and aggregated observation features
        (``{column}_{agg}``) when metric and aggregation names overlap.

        :param profile_frames: Feature frames keyed by ``row_id``.
        :type profile_frames: list
        :param aggregated_frames: Aggregated frames keyed by the profile keys.
        :type aggregated_frames: list
        :raises ValueError: Naming the duplicated columns.
        """
        key_columns = {"row_id", *PROFILE_KEYS}
        seen: set = set()
        duplicated: set = set()
        for frame in [*profile_frames, *aggregated_frames]:
            for col in frame.columns:
                if col in key_columns:
                    continue
                if col in seen:
                    duplicated.add(col)
                seen.add(col)
        if duplicated:
            raise ValueError(
                f"Multiple features produce the same output column(s) "
                f"{sorted(duplicated)}. Rename or drop the overlapping "
                f"summary statistic / aggregation so each column is unique."
            )

    def _feature_level(self, feature_info: Dict) -> str:
        """
        Return the declared level of the feature named in ``feature_info``.

        :param feature_info: A dictionary of feature extraction parameters.
        :type feature_info: Dict
        :return: ``"observation"`` or ``"profile"``.
        :rtype: str
        :raises ValueError: If the feature name is missing or unknown.
        """
        feature_name = feature_info.get("feature")
        feature_class = FEATURE_REGISTRY.get(feature_name)
        if feature_class is None:
            raise ValueError(f"Unknown feature '{feature_name}' in feature params.")
        return feature_class.level

    def _extract_aggregated_features(
        self, target_name: str, feature_info: Dict
    ) -> pl.DataFrame:
        """
        Run an observation-level feature on the target's observation rows and
        aggregate its columns per profile.

        :param target_name: The target for which features will be extracted.
        :type target_name: str
        :param feature_info: A dictionary of feature extraction parameters;
                             must contain a non-empty ``agg`` list.
        :type feature_info: Dict
        :return: One row per profile: the profile keys plus one
                 ``{column}_{agg}`` column per (feature column, aggregation).
        :rtype: pl.DataFrame
        :raises ValueError: If ``agg`` is missing/empty, an aggregation name is
                            unknown, or :attr:`observation_rows` lacks the target.
        """
        feature_name = feature_info.get("feature")
        agg_names = feature_info.get("agg")
        if not agg_names:
            raise ValueError(
                f"Feature '{feature_name}' is observation-level and cannot be "
                f"used directly at profile level; add 'agg: [...]' to its "
                f"feature params or remove it."
            )
        validate_aggregation_names(feature_name, agg_names)

        if self.observation_rows is None or target_name not in self.observation_rows:
            raise ValueError(
                f"Observation rows for target '{target_name}' are required to "
                f"aggregate feature '{feature_name}' but were not provided. "
                f"Use a profile-level locate step (e.g. 'LocateDataSetProfile')."
            )
        observation_rows = self.observation_rows[target_name]

        ds = load_feature_class(
            target_name,
            feature_info,
            self.selected_profiles,
            self.filtered_input,
            {target_name: observation_rows},
            self.summary_stats,
        )
        ds.scale_first()
        ds.extract_features()
        ds.scale_second()

        feature_columns = [col for col in ds.features.columns if col != "row_id"]
        return (
            observation_rows.select(["row_id", *PROFILE_KEYS])
            .join(ds.features, on="row_id", maintain_order="left")
            .group_by(PROFILE_KEYS)
            .agg(build_aggregation_exprs(feature_columns, agg_names))
        )
