"""
This module provides a Random Forest regressor wrapper, inheriting from
:class:`aiqclib.common.base.scikit_learn_regressor_base.SklearnRegressorModelBase`.

It trains :class:`sklearn.ensemble.RandomForestRegressor` on proportion
labels (floats in [0, 1]), as produced by the profile-level pipeline's
``label_mode: proportion``.
"""

from typing import Dict, Any

from sklearn import ensemble

from aiqclib.common.base.config_base import ConfigBase
from aiqclib.common.base.scikit_learn_regressor_base import SklearnRegressorModelBase


class RandomForestRegressor(SklearnRegressorModelBase):
    """
    A Random Forest regressor wrapper class for training and testing on
    proportion labels.

    Mirrors :class:`~aiqclib.train.models.random_forest.RandomForest` but
    uses :class:`sklearn.ensemble.RandomForestRegressor`.

    .. note::
       This class sets :attr:`expected_class_name` to
       ``"RandomForestRegressor"`` and :attr:`short_name` to ``"RFR"``.
    """

    expected_class_name: str = "RandomForestRegressor"
    short_name: str = "RFR"

    def __init__(self, config: ConfigBase) -> None:
        """
        Initialize the Random Forest regressor with default or user-specified
        parameters.

        :param config: A configuration object providing model parameters.
        :type config: aiqclib.common.base.config_base.ConfigBase
        """
        super().__init__(config=config)

        self.model_params: Dict[str, Any] = {
            "n_estimators": 100,
            "n_jobs": -1,
        }
        # Update model parameters with config step parameters
        model_params = self.config.get_model_params(
            self.expected_class_name, self.short_name
        )
        self.model_params.update(model_params)

    def _get_model_class(self) -> Any:
        """
        Return the Random Forest regressor class.

        :return: The RandomForestRegressor class.
        :rtype: type[sklearn.ensemble.RandomForestRegressor]
        """
        return ensemble.RandomForestRegressor
