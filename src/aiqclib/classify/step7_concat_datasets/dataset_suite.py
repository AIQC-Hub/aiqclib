"""
This module provides the ConcatDataSetSuite class, which is responsible for merging
multi-method model predictions into a wide-format dataset aligned with the
original input data.
"""

from typing import Optional, Dict

import polars as pl

from aiqclib.classify.step7_concat_datasets.concat_base import ConcatDatasetsBase
from aiqclib.common.base.config_base import ConfigBase


class ConcatDataSetSuite(ConcatDatasetsBase):
    """
    A subclass of :class:`ConcatDatasetsBase` to concatenate multi-method predictions
    and the input dataset.

    This class handles predictions containing a 'method' column, expanding them
    into a wide format where each method's predictions and scores become separate
    columns formatted as ``{method}_{target}_predicted`` and ``{method}_{target}_score``.

    :ivar expected_class_name: The name of the class used for validation or logging.
    :vartype expected_class_name: str
    """

    expected_class_name: str = "ConcatDataSetSuite"

    def __init__(
        self,
        config: ConfigBase,
        input_data: Optional[pl.DataFrame] = None,
        predictions: Optional[Dict[str, pl.DataFrame]] = None,
    ) -> None:
        """
        Initialize the concatenation workflow for multi-method predictions and input data.

        :param config: A dataset configuration object that manages paths, target
            definitions, and parameters for data processing.
        :type config: ConfigBase
        :param input_data: A Polars DataFrame containing all available data to which
            predictions will be concatenated, defaults to None.
        :type input_data: Optional[pl.DataFrame]
        :param predictions: A dictionary mapping each target to its respective Polars
            DataFrame of predictions (containing a 'method' column), defaults to None.
        :type predictions: Optional[Dict[str, pl.DataFrame]]
        :return: None
        :rtype: None
        """
        super().__init__(
            config=config,
            input_data=input_data,
            predictions=predictions,
        )

    def merge_predictions(self) -> None:
        """
        Merges the input data with the multi-method predictions for each target into
        a single wide Polars DataFrame.

        The method pivots the 'method' column into distinct prediction and score columns
        for each algorithm. It uses the following column naming convention:

          - ``{key}_label``
          - ``{method}_{key}_predicted``
          - ``{method}_{key}_score``

        The result is stored in the :attr:`merged_predictions` attribute.

        :raises ValueError: If :attr:`predictions` or :attr:`input_data` is None.
        :return: None
        :rtype: None
        """
        if self.input_data is None:
            raise ValueError("Member variable 'input_data' must not be empty.")

        if self.predictions is None:
            raise ValueError("Member variable 'predictions' must not be empty.")

        join_keys = ["platform_code", "profile_no", "observation_no"]

        # Start with the original input data as the base
        merged_df = self.input_data

        for key, df in self.predictions.items():
            # 1. Extract the ground truth label (identical across methods for the same observation)
            target_wide = df.select(join_keys + ["label"]).unique(
                subset=join_keys, keep="first"
            )
            target_wide = target_wide.rename({"label": f"{key}_label"})

            # 2. Extract unique methods present in this target's predictions
            methods = df["method"].unique().to_list()

            # 3. For each method, isolate its rows, rename the columns, and join to target_wide
            for m in methods:
                m_df = (
                    df.filter(pl.col("method") == m)
                    .select(join_keys + ["predicted_label", "score"])
                    .rename(
                        {
                            "predicted_label": f"{m.lower()}_{key}_predicted",
                            "score": f"{m.lower()}_{key}_score",
                        }
                    )
                )

                target_wide = target_wide.join(m_df, on=join_keys, how="left")

            # 4. Join this fully widened target dataframe to the main merged dataframe
            merged_df = merged_df.join(target_wide, on=join_keys, how="left")

        self.merged_predictions = merged_df
