"""
This module provides an XGBoost model wrapper, inheriting from
:class:`aiqclib.common.base.scikit_learn_model_base.SklearnModelBase`.

It facilitates training, prediction, and evaluation of an XGBoost classifier using Polars DataFrames,
converting them to Pandas for compatibility with the `xgboost` library.
"""

from typing import Dict, Any

import xgboost as xgb

from aiqclib.common.base.config_base import ConfigBase
from aiqclib.common.base.scikit_learn_model_base import SklearnModelBase


class XGBoost(SklearnModelBase):
    """
    An XGBoost model wrapper class for training and testing.

    Inherits from :class:`aiqclib.common.base.scikit_learn_model_base.SklearnModelBase`
    to reuse common Scikit-Learn API logic.

    Features include:

    - Automatic application of ``model_params`` from the YAML config, if defined;
      otherwise, uses default hyperparameters.
    - Uses :class:`xgboost.XGBClassifier`.

    .. note::
       This class sets :attr:`expected_class_name` to ``"XGBoost"``
       and :attr:`short_name` to ``"XGB"``.
    """

    expected_class_name: str = "XGBoost"
    short_name: str = "XGB"

    def __init__(self, config: ConfigBase) -> None:
        """
        Initialize the XGBoost model with default or user-specified parameters.

        :param config: A configuration object providing model parameters.
        :type config: aiqclib.common.base.config_base.ConfigBase
        """
        super().__init__(config=config)

        self.model_params: Dict[str, Any] = {
            "n_estimators": 100,
            "max_depth": 10,
            "learning_rate": 0.1,
            "eval_metric": "logloss",
            "scale_pos_weight": 1,
            "n_jobs": -1,
        }
        # Update model parameters with config step parameters
        model_params = self.config.get_model_params(
            self.expected_class_name, self.short_name
        )
        self.model_params.update(model_params)

    def _get_model_class(self) -> Any:
        """
        Return the XGBoost classifier class.

        :return: The XGBClassifier class.
        :rtype: type[xgboost.XGBClassifier]
        """
        return xgb.XGBClassifier
