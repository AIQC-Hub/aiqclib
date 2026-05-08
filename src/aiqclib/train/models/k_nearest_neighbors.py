"""
This module provides a K-Nearest Neighbors (KNN) model wrapper, inheriting
from `aiqclib.common.base.scikit_learn_model_base.SklearnModelBase`.

It facilitates training, prediction, and evaluation of a KNN classifier using
Polars DataFrames, converting them to Pandas for compatibility with the
`sklearn` library.
"""

from typing import Dict, Any

from sklearn.neighbors import KNeighborsClassifier as SklearnKNN

from aiqclib.common.base.config_base import ConfigBase
from aiqclib.common.base.scikit_learn_model_base import SklearnModelBase


class KNearestNeighbors(SklearnModelBase):
    """
    A K-Nearest Neighbors (KNN) model wrapper class for training and testing.

    Inherits from :class:`SklearnModelBase` to reuse common Scikit-Learn API logic.

    Features include:

    - Automatic application of ``model_params`` from the YAML config, if defined;
      otherwise, uses default hyperparameters.
    - Uses ``sklearn.neighbors.KNeighborsClassifier``.

    .. note::
       This class sets :attr:`expected_class_name` to ``"KNearestNeighbors"``.
       Note that KNN can be computationally expensive during prediction for large datasets.
    """

    expected_class_name: str = "KNearestNeighbors"
    short_name: str = "KNN"

    def __init__(self, config: ConfigBase) -> None:
        """
        Initialize the KNN model with default or user-specified parameters.

        :param config: A configuration object providing model parameters.
        :type config: ConfigBase
        """
        super().__init__(config=config)

        self.model_params: Dict[str, Any] = {
            "n_neighbors": 5,
            "weights": "uniform",
            "algorithm": "auto",
            "leaf_size": 30,
            "p": 2,  # Euclidean distance
            "metric": "minkowski",
            "n_jobs": -1,
        }
        # Update model parameters with config step parameters
        model_params = self.config.get_model_params(
            self.expected_class_name, self.short_name
        )
        self.model_params.update(model_params)
        self.allow_na = False

    def _get_model_class(self) -> Any:
        """
        Return the Scikit-Learn KNeighborsClassifier class.

        :returns: The KNeighborsClassifier class.
        :rtype: Any
        """
        return SklearnKNN
