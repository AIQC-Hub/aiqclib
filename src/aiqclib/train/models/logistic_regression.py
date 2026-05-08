"""
This module provides a Logistic Regression model wrapper, inheriting from `aiqclib.common.base.scikit_learn_model_base.SklearnModelBase`.

It facilitates training, prediction, and evaluation of a Scikit-Learn Logistic Regression classifier
using Polars DataFrames.
"""

from typing import Dict, Any

from sklearn.linear_model import LogisticRegression as SklearnLR

from aiqclib.common.base.config_base import ConfigBase
from aiqclib.common.base.scikit_learn_model_base import SklearnModelBase


class LogisticRegression(SklearnModelBase):
    """
    A Logistic Regression model wrapper class for training and testing.

    Inherits from :class:`SklearnModelBase` to reuse common Scikit-Learn API logic.

    Features include:

    - Automatic application of ``model_params`` from the YAML config, if defined;
      otherwise, uses default hyperparameters suitable for standard classification tasks.
    - Uses ``sklearn.linear_model.LogisticRegression``.

    .. note::
       This class sets :attr:`expected_class_name` to ``"LogisticRegression"``.
    """

    expected_class_name: str = "LogisticRegression"
    short_name: str = "Logit"

    def __init__(self, config: ConfigBase) -> None:
        """
        Initialize the Logistic Regression model with default or user-specified parameters.

        :param config: A configuration object providing model parameters.
        :type config: ConfigBase
        """
        super().__init__(config=config)

        # Default parameters for Logistic Regression
        self.model_params: Dict[str, Any] = {
            "penalty": "l2",
            "C": 1.0,
            "solver": "lbfgs",
            "class_weight": "balanced",
            "max_iter": 200,
        }
        # Update model parameters with config step parameters
        model_params = self.config.get_model_params(
            self.expected_class_name, self.short_name
        )
        self.model_params.update(model_params)
        self.allow_na = False

    def _get_model_class(self) -> Any:
        """
        Return the Scikit-Learn LogisticRegression class.

        :return: The LogisticRegression class.
        :rtype: Any
        """
        return SklearnLR
