"""
This module provides a Decision Tree model wrapper, inheriting
from `aiqclib.common.base.scikit_learn_model_base.SklearnModelBase`.

It facilitates training, prediction, and evaluation of a Decision Tree classifier using
Polars DataFrames, converting them to Pandas for compatibility with the
`sklearn` library.
"""

from typing import Dict, Any

from sklearn.tree import DecisionTreeClassifier as SklearnDT

from aiqclib.common.base.config_base import ConfigBase
from aiqclib.common.base.scikit_learn_model_base import SklearnModelBase


class DecisionTree(SklearnModelBase):
    """
    A Decision Tree model wrapper class for training and testing.

    Inherits from :class:`SklearnModelBase` to reuse common Scikit-Learn API logic.

    Features include:

    - Automatic application of ``model_params`` from the YAML config, if defined;
      otherwise, uses default hyperparameters.
    - Uses ``sklearn.tree.DecisionTreeClassifier``.

    .. note::
       This class sets :attr:`expected_class_name` to ``"DecisionTree"``.
       Single Decision Trees in scikit-learn generally do not support the ``n_jobs`` parameter.
    """

    expected_class_name: str = "DecisionTree"
    short_name: str = "DT"

    def __init__(self, config: ConfigBase) -> None:
        """
        Initialize the Decision Tree model with default or user-specified parameters.

        :param config: A configuration object providing model parameters.
        :type config: aiqclib.common.base.config_base.ConfigBase
        """
        super().__init__(config=config)

        self.model_params: Dict[str, Any] = {
            "criterion": "gini",
            "splitter": "best",
            "max_depth": 10,
            "min_samples_split": 10,
            "min_samples_leaf": 5,
            "max_features": None,
            "random_state": None,
            "class_weight": "balanced",
            "ccp_alpha": 0.001,
        }
        # Update model parameters with config step parameters
        model_params = self.config.get_model_params(
            self.expected_class_name, self.short_name
        )
        self.model_params.update(model_params)

    def _get_model_class(self) -> Any:
        """
        Return the Scikit-Learn DecisionTreeClassifier class.

        :return: The DecisionTreeClassifier class.
        :rtype: typing.Any
        """
        return SklearnDT
