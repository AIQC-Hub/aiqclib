"""
This module provides a Linear Discriminant Analysis (LDA) model wrapper, inheriting
from `aiqclib.common.base.scikit_learn_model_base.SklearnModelBase`.

It facilitates training, prediction, and evaluation of an LDA classifier using
Polars DataFrames, converting them to Pandas for compatibility with the
`sklearn` library.
"""

from typing import Dict, Any

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as SklearnLDA

from aiqclib.common.base.config_base import ConfigBase
from aiqclib.common.base.scikit_learn_model_base import SklearnModelBase


class LinearDiscriminantAnalysis(SklearnModelBase):
    """
    A Linear Discriminant Analysis (LDA) model wrapper class for training and testing.

    Inherits from :class:`SklearnModelBase` to reuse common Scikit-Learn API logic.

    Features include:

    - Automatic application of ``model_params`` from the YAML config, if defined;
      otherwise, uses default hyperparameters.
    - Uses ``sklearn.discriminant_analysis.LinearDiscriminantAnalysis``.

    .. note::
       This class sets :attr:`expected_class_name` to ``"LinearDiscriminantAnalysis"``.
       Note that LDA in scikit-learn does not support the ``n_jobs`` parameter.
    """

    expected_class_name: str = "LinearDiscriminantAnalysis"
    short_name: str = "LDA"

    def __init__(self, config: ConfigBase) -> None:
        """
        Initialize the LDA model with default or user-specified parameters.

        :param config: A configuration object providing model parameters.
        :type config: aiqclib.common.base.config_base.ConfigBase
        """
        super().__init__(config=config)

        self.model_params: Dict[str, Any] = {
            "solver": "svd",
            "shrinkage": None,
            "priors": None,
            "n_components": None,
            "store_covariance": False,
            "tol": 1.0e-4,
        }
        # Update model parameters with config step parameters
        model_params = self.config.get_model_params(
            self.expected_class_name, self.short_name
        )
        self.model_params.update(model_params)
        self.allow_na = False

    def _get_model_class(self) -> Any:
        """
        Return the Scikit-Learn LinearDiscriminantAnalysis class.

        :return: The LinearDiscriminantAnalysis class from scikit-learn.
        :rtype: type
        """
        return SklearnLDA
