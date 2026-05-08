"""
This module provides a Multi-layer Perceptron (MLP) model wrapper, inheriting
from `aiqclib.common.base.scikit_learn_model_base.SklearnModelBase`.

It facilitates training, prediction, and evaluation of an MLP classifier using
Polars DataFrames, converting them to Pandas for compatibility with the
`sklearn` library.
"""

from typing import Dict, Any

from sklearn.neural_network import MLPClassifier as SklearnMLP

from aiqclib.common.base.config_base import ConfigBase
from aiqclib.common.base.scikit_learn_model_base import SklearnModelBase


class MultilayerPerceptron(SklearnModelBase):
    """
    A Multi-layer Perceptron (MLP) model wrapper class for training and testing.

    Inherits from :class:`~aiqclib.common.base.scikit_learn_model_base.SklearnModelBase`
    to reuse common Scikit-Learn API logic.

    Features include:

    - Automatic application of ``model_params`` from the YAML config, if defined;
      otherwise, uses default hyperparameters.
    - Uses :class:`sklearn.neural_network.MLPClassifier`.

    .. note::
       This class sets :attr:`expected_class_name` to ``"MultilayerPerceptron"``.
       This is a feedforward neural network implementation.
    """

    expected_class_name: str = "MultilayerPerceptron"
    short_name: str = "MLP"

    def __init__(self, config: ConfigBase) -> None:
        """
        Initialize the MLP model with default or user-specified parameters.

        :param config: A configuration object providing model parameters.
        :type config: aiqclib.common.base.config_base.ConfigBase
        """
        super().__init__(config=config)

        self.model_params: Dict[str, Any] = {
            "hidden_layer_sizes": (50,),
            "activation": "relu",
            "solver": "adam",
            "alpha": 0.0001,
            "batch_size": "auto",
            "learning_rate": "constant",
            "learning_rate_init": 0.001,
            "max_iter": 100,
            "shuffle": True,
            "random_state": None,
            "tol": 1e-3,
            "early_stopping": True,
            "n_iter_no_change": 5,
        }
        # Update model parameters with config step parameters
        model_params = self.config.get_model_params(
            self.expected_class_name, self.short_name
        )
        self.model_params.update(model_params)
        self.allow_na = False

    def _get_model_class(self) -> Any:
        """
        Return the Scikit-Learn MLPClassifier class.

        :return: The MLPClassifier class.
        :rtype: Any
        """
        return SklearnMLP
