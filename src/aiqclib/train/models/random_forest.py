"""
This module provides a Random Forest model wrapper, inheriting
from `aiqclib.common.base.scikit_learn_model_base.SklearnModelBase`.

It facilitates training, prediction, and evaluation of a Random Forest classifier using
Polars DataFrames, converting them to Pandas for compatibility with the
`sklearn` library.
"""

from typing import Dict, Any

from sklearn.ensemble import RandomForestClassifier as SklearnRF

from aiqclib.common.base.config_base import ConfigBase
from aiqclib.common.base.scikit_learn_model_base import SklearnModelBase


class RandomForest(SklearnModelBase):
    """
    A Random Forest model wrapper class for training and testing.

    Inherits from :class:`SklearnModelBase` to reuse common Scikit-Learn API logic.

    Features include:

    - Automatic application of ``model_params`` from the YAML config, if defined;
      otherwise, uses default hyperparameters.
    - Uses ``sklearn.ensemble.RandomForestClassifier``.

    .. note::
       This class sets :attr:`expected_class_name` to ``"RandomForest"``.
    """

    expected_class_name: str = "RandomForest"
    short_name: str = "RF"

    def __init__(self, config: ConfigBase) -> None:
        """
        Initialize the Random Forest model with default or user-specified parameters.

        :param config: A configuration object providing model parameters.
        :type config: ConfigBase
        """
        super().__init__(config=config)

        self.model_params: Dict[str, Any] = {
            "n_estimators": 100,
            "criterion": "gini",
            "max_depth": 10,
            "min_samples_split": 10,
            "min_samples_leaf": 5,
            "max_features": "sqrt",
            "bootstrap": True,
            "n_jobs": -1,
            "random_state": None,
            "class_weight": "balanced_subsample",
        }
        # Update model parameters with config step parameters
        model_params = self.config.get_model_params(
            self.expected_class_name, self.short_name
        )
        self.model_params.update(model_params)

    def _get_model_class(self) -> Any:
        """
        Return the Scikit-Learn RandomForestClassifier class.

        :return: The RandomForestClassifier class.
        :rtype: Any
        """
        return SklearnRF
