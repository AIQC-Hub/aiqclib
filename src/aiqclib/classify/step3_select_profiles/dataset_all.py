"""
This module defines the `SelectDataSetAll` class, a specialized profile selection
mechanism within the aiqclib library. It is designed to select all available
profiles from a given input dataset (typically Copernicus CTD data) and
assign initial labels and identifiers for subsequent classification tasks.
"""

from typing import Optional, List

import polars as pl

from aiqclib.common.base.config_base import ConfigBase
from aiqclib.prepare.step3_select_profiles.select_base import ProfileSelectionBase


class SelectDataSetAll(ProfileSelectionBase):
    """
    A subclass of :class:`ProfileSelectionBase` that selects all profiles from
    Copernicus CTD data.

    This class initializes a selection process where all input profiles are
    considered, and initial labels (e.g., 'negative') and unique identifiers
    are assigned before further processing or classification.
    """

    expected_class_name: str = "SelectDataSetAll"

    def __init__(
        self, config: ConfigBase, input_data: Optional[pl.DataFrame] = None
    ) -> None:
        """
        Initialize an instance for selecting and labeling profiles.

        :param config: The configuration object specifying paths and parameters for the selection process.
        :type config: aiqclib.common.base.config_base.ConfigBase
        :param input_data: An optional Polars DataFrame of all profiles from which negative and positive examples are to be selected.
        :type input_data: polars.DataFrame, optional
        """
        super().__init__(config=config, input_data=input_data)

        #: Default file name to which selected profiles are written.
        self.default_file_name: str = "selected_profiles_classify.parquet"

        #: Full path for the output file, resolved via the config.
        self.output_file_name: str = self.config.get_full_file_name(
            step_name="select", default_file_name=self.default_file_name
        )

        #: Columns used as unique identifiers for grouping/merging
        #: (e.g., by platform or profile).
        self.key_col_names: List[str] = [
            "platform_code",
            "profile_no",
            "profile_timestamp",
            "longitude",
            "latitude",
        ]

    def select_all_profiles(self) -> None:
        """
        Select all profiles from the input data and prepare them with initial
        labeling and unique identifiers.

        This method processes the :attr:`input_data` to create a DataFrame
        of unique profiles. It adds the following columns:

        - ``neg_profile_id`` (uint32): Initialized to 0. This column can
          serve as a placeholder for later assignment of specific negative
          profile identifiers, though it is not a unique ID in this step.
        - ``label`` (uint32): Initialized to 0, indicating an unclassified
          or 'negative' profile in the context of subsequent classification.
        - ``profile_id`` (int): A unique 1-based row index assigned to
          each selected profile, serving as its primary identifier.

        The resulting DataFrame is assigned to :attr:`selected_profiles`.
        All profiles are made unique based on their key columns
        (platform, profile number, timestamp, longitude, latitude)
        before `profile_id` is assigned.
        """
        self.selected_profiles = (
            self.input_data.with_columns(
                pl.lit(0, dtype=pl.UInt32).alias("neg_profile_id"),
                pl.lit(0, dtype=pl.UInt32).alias("label"),
            )
            .select(
                pl.col("platform_code"),
                pl.col("profile_no"),
                pl.col("profile_timestamp"),
                pl.col("longitude"),
                pl.col("latitude"),
                pl.col("neg_profile_id"),
                pl.col("label"),
            )
            .unique(maintain_order=True)
            .with_row_index("profile_id", offset=1)
        )

    def label_profiles(self) -> None:
        """
        Select and label positive and negative datasets before combining them
        into a single DataFrame in :attr:`selected_profiles`.

        In this specific implementation, all profiles are initially selected
        and labeled as 'negative' (label 0) by calling
        :meth:`select_all_profiles`. This method effectively serves as the
        entry point for the profile selection and initial labeling process.
        """
        self.select_all_profiles()
