"""
This module provides an XGBoost regressor wrapper, inheriting from
:class:`aiqclib.common.base.scikit_learn_regressor_base.SklearnRegressorModelBase`.

It trains :class:`xgboost.XGBRegressor` on proportion labels (floats in
[0, 1]), as produced by the profile-level pipeline's
``label_mode: proportion``.
"""

from typing import Dict, Any

import xgboost as xgb

from aiqclib.common.base.config_base import ConfigBase
from aiqclib.common.base.scikit_learn_regressor_base import SklearnRegressorModelBase


class XGBoostRegressor(SklearnRegressorModelBase):
    """
    An XGBoost regressor wrapper class for training and testing on
    proportion labels.

    Mirrors :class:`~aiqclib.train.models.xgboost.XGBoost` but uses
    :class:`xgboost.XGBRegressor` with an RMSE objective metric and no
    class-weighting parameter.

    .. note::
       This class sets :attr:`expected_class_name` to ``"XGBoostRegressor"``
       and :attr:`short_name` to ``"XGBR"``.
    """

    expected_class_name: str = "XGBoostRegressor"
    short_name: str = "XGBR"

    def __init__(self, config: ConfigBase) -> None:
        """
        Initialize the XGBoost regressor with default or user-specified parameters.

        :param config: A configuration object providing model parameters.
        :type config: aiqclib.common.base.config_base.ConfigBase
        """
        super().__init__(config=config)

        self.model_params: Dict[str, Any] = {
            "n_estimators": 100,
            "max_depth": 10,
            "learning_rate": 0.1,
            "eval_metric": "rmse",
            "n_jobs": -1,
        }
        # Update model parameters with config step parameters
        model_params = self.config.get_model_params(
            self.expected_class_name, self.short_name
        )
        self.model_params.update(model_params)

    def _get_model_class(self) -> Any:
        """
        Return the XGBoost regressor class.

        :return: The XGBRegressor class.
        :rtype: type[xgboost.XGBRegressor]
        """
        return xgb.XGBRegressor
