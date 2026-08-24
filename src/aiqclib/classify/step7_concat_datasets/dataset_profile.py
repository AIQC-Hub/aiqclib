"""
This module defines the ConcatDataSetProfile class, which merges profile-level
model predictions into a one-row-per-profile output.

Unlike the observation-level :class:`ConcatDataSetAll`, the predictions carry
no ``observation_no``; they are joined on the profile keys onto per-profile
metadata derived from the input (timestamp and position). The optional
``broadcast_to_observations`` step parameter instead joins the predictions
onto the full observation-level input, repeating each profile's prediction
for every observation of that profile.
"""

from typing import Optional, Dict

import polars as pl

from aiqclib.classify.step7_concat_datasets.concat_base import ConcatDatasetsBase
from aiqclib.common.base.config_base import ConfigBase
from aiqclib.common.constants import PROFILE_KEYS


class ConcatDataSetProfile(ConcatDatasetsBase):
    """
    A subclass of :class:`ConcatDatasetsBase` that merges profile-level
    predictions into one row per profile.

    Output columns: the profile keys plus ``profile_timestamp``,
    ``longitude`` and ``latitude`` from the input, and per target the
    renamed ``{target}_label`` / ``{target}_predicted`` / ``{target}_score``
    columns. With the ``concat`` step parameter
    ``broadcast_to_observations: true`` the predictions are joined onto the
    full observation-level input instead.
    """

    expected_class_name: str = "ConcatDataSetProfile"

    #: Input columns describing a profile, kept in the per-profile output.
    profile_metadata_columns = ["profile_timestamp", "longitude", "latitude"]

    def __init__(
        self,
        config: ConfigBase,
        input_data: Optional[pl.DataFrame] = None,
        predictions: Optional[Dict[str, pl.DataFrame]] = None,
    ) -> None:
        """
        Initialize the profile-level concatenation workflow.

        :param config: A classification configuration object.
        :type config: ConfigBase
        :param input_data: A Polars DataFrame containing all available data.
        :type input_data: Optional[pl.DataFrame]
        :param predictions: A dictionary mapping each target to its
                            per-profile predictions.
        :type predictions: Optional[Dict[str, pl.DataFrame]]
        """
        super().__init__(
            config=config,
            input_data=input_data,
            predictions=predictions,
        )

        #: One row per profile by default.
        self.default_file_name: str = "predictions_profile.parquet"
        self.output_file_name: str = self.config.get_full_file_name(
            step_name="concat", default_file_name=self.default_file_name
        )

        #: When True, join the per-profile predictions back onto the full
        #: observation-level input instead of the per-profile metadata.
        self.broadcast_to_observations: bool = bool(
            self.config.get_step_params("concat").get(
                "broadcast_to_observations", False
            )
        )

    def merge_predictions(self) -> None:
        """
        Merge the per-target profile predictions into a single DataFrame.

        The per-target prediction frames are renamed to
        ``{target}_label`` / ``{target}_predicted`` / ``{target}_score``,
        aligned on the profile keys, and joined onto either the per-profile
        metadata (default) or the full observation-level input
        (``broadcast_to_observations``).

        :raises ValueError: If :attr:`predictions` or :attr:`input_data` is None.
        """
        if self.input_data is None:
            raise ValueError("Member variable 'input_data' must not be empty.")

        if self.predictions is None:
            raise ValueError("Member variable 'predictions' must not be empty.")

        if self.broadcast_to_observations:
            left = self.input_data
        else:
            metadata_columns = [
                col
                for col in self.profile_metadata_columns
                if col in self.input_data.columns
            ]
            left = (
                self.input_data.select(PROFILE_KEYS + metadata_columns)
                .unique(subset=PROFILE_KEYS)
                .sort(PROFILE_KEYS)
            )

        self.merged_predictions = left.join(
            pl.concat(
                [
                    df.rename(
                        {
                            "label": f"{key}_label",
                            "predicted_label": f"{key}_predicted",
                            "score": f"{key}_score",
                        }
                    ).select(
                        PROFILE_KEYS
                        + [f"{key}_label", f"{key}_predicted", f"{key}_score"]
                    )
                    for key, df in self.predictions.items()
                ],
                how="align",
            ),
            on=PROFILE_KEYS,
        )
