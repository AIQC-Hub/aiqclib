"""
Module for concatenating input datasets with machine learning model predictions.

This module provides the :class:`ConcatDatasetsBase` class, which handles the logic
of merging original input features with generated predictions from one or more
targets and saving the result to a Parquet file.
"""

import os
from typing import Dict, Optional

import polars as pl

from aiqclib.common.base.config_base import ConfigBase
from aiqclib.common.base.dataset_base import DataSetBase


class ConcatDatasetsBase(DataSetBase):
    """
    Abstract base class for concatenating predictions and the original dataset.

    Inherits from :class:`~aiqclib.common.base.dataset_base.DataSetBase` to ensure configuration consistency.
    The concatenated dataset, once generated, can be written to Parquet files.
    """

    def __init__(
        self,
        config: ConfigBase,
        input_data: Optional[pl.DataFrame] = None,
        predictions: Optional[Dict[str, pl.DataFrame]] = None,
    ) -> None:
        """
        Initialize the dataset concatenation base class.

        :param config: The configuration object, containing paths and target definitions.
        :type config: aiqclib.common.base.config_base.ConfigBase
        :param input_data: A Polars DataFrame providing the full dataset to which
                           predictions will be concatenated, defaults to None.
        :type input_data: Optional[polars.DataFrame]
        :param predictions: A dictionary mapping each target to its respective
                            subset of predictions, defaults to None.
        :type predictions: Optional[Dict[str, polars.DataFrame]]
        :raises NotImplementedError: If the subclass does not define
                                     ``expected_class_name``.
        :raises ValueError: If the provided YAML config does not match this class's
                            ``expected_class_name``.
        :return: None
        :rtype: NoneType
        """
        super().__init__(step_name="concat", config=config)

        #: The default pattern to use when writing feature files for each target.
        self.default_file_name: str = "predictions.parquet"

        #: Output file name to store the concatenated dataset
        self.output_file_name: str = self.config.get_full_file_name(
            step_name="concat", default_file_name=self.default_file_name
        )

        self.input_data: Optional[pl.DataFrame] = input_data

        #: A dict of Polars DataFrames, one per target, containing classification results.
        self.predictions: Optional[Dict[str, pl.DataFrame]] = predictions

        self.merged_predictions: Optional[pl.DataFrame] = None

    def merge_predictions(self) -> None:
        """
        Merges the input data with the predictions for each target into a single Polars DataFrame.

        The method concatenates individual prediction DataFrames (one per target)
        and then joins them with the original input data based on common
        identifier columns ('platform_code', 'profile_no', 'observation_no').
        The 'label', 'predicted_label', and 'score' columns from each target's predictions are
        renamed to include the target key (e.g., 'targetA_label', 'targetA_predicted')
        to avoid name collisions.

        The result is stored in the :attr:`merged_predictions` attribute.

        :raises ValueError: If :attr:`predictions` or :attr:`input_data` is None when this method is called.
        :return: None
        :rtype: NoneType
        """
        if self.input_data is None:
            raise ValueError("Member variable 'input_data' must not be empty.")

        if self.predictions is None:
            raise ValueError("Member variable 'predictions' must not be empty.")

        self.merged_predictions = self.input_data.join(
            pl.concat(
                [
                    df.rename(
                        {
                            "label": f"{key}_label",
                            "predicted_label": f"{key}_predicted",
                            "score": f"{key}_score",
                        }
                    ).select(
                        [
                            "platform_code",
                            "profile_no",
                            "observation_no",
                            f"{key}_label",
                            f"{key}_predicted",
                            f"{key}_score",
                        ]
                    )
                    for key, df in self.predictions.items()
                ],
                how="align",
            ),
            on=["platform_code", "profile_no", "observation_no"],
        )

    def write_merged_predictions(self) -> None:
        """
        Writes the merged predictions DataFrame to a Parquet file.

        The output directory is created if it does not exist.
        The file path is determined by :attr:`output_file_name`.

        :raises ValueError: If :attr:`merged_predictions` is None when this method is called.
        :return: None
        :rtype: NoneType
        """
        if self.merged_predictions is None:
            raise ValueError("Member variable 'merged_predictions' must not be empty.")

        os.makedirs(os.path.dirname(self.output_file_name), exist_ok=True)
        self.merged_predictions.write_parquet(self.output_file_name)
