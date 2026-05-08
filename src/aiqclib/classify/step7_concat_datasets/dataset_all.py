"""
This module defines the ConcatDataSetAll class, which extends ConcatDatasetsBase
to facilitate the concatenation of model predictions with the original input dataset.
It is designed to integrate into a larger data quality control (DQC) workflow,
specifically within the classification and merging steps.
"""

from typing import Optional, Dict

import polars as pl

from aiqclib.classify.step7_concat_datasets.concat_base import ConcatDatasetsBase
from aiqclib.common.base.config_base import ConfigBase


class ConcatDataSetAll(ConcatDatasetsBase):
    """
    A subclass of :class:`ConcatDatasetsBase` to concatenate predictions and the input dataset.

    This class sets its :attr:`expected_class_name` to ``"ConcatDataSetAll"``,
    ensuring it is recognized in the YAML configuration as a valid
    processing class. It inherits the concatenation pipeline from
    :class:`ConcatDatasetsBase`.

    :ivar expected_class_name: The identifier used for configuration mapping.
    :vartype expected_class_name: str
    """

    expected_class_name: str = "ConcatDataSetAll"

    def __init__(
        self,
        config: ConfigBase,
        input_data: Optional[pl.DataFrame] = None,
        predictions: Optional[Dict[str, pl.DataFrame]] = None,
    ) -> None:
        """
        Initialize the concatenation workflow for predictions and input data.

        :param config: A dataset configuration object that manages paths,
                       target definitions, and parameters for data processing.
        :type config: ConfigBase
        :param input_data: A Polars DataFrame containing all available data
                           to which predictions will be concatenated, defaults to None.
        :type input_data: Optional[pl.DataFrame]
        :param predictions: A dictionary mapping each target (e.g., a classification task)
                            to its respective Polars DataFrame of predictions,
                            defaults to None.
        :type predictions: Optional[Dict[str, pl.DataFrame]]
        """
        super().__init__(
            config=config,
            input_data=input_data,
            predictions=predictions,
        )
