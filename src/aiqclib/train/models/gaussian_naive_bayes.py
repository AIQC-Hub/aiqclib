"""
This module provides a Gaussian Naive Bayes model wrapper, inheriting
from `aiqclib.common.base.scikit_learn_model_base.SklearnModelBase`.

It facilitates training, prediction, and evaluation of a Naive Bayes classifier using
Polars DataFrames, converting them to Pandas for compatibility with the
`sklearn` library.
"""

from typing import Dict, Any

from sklearn.naive_bayes import GaussianNB as SklearnGNB

from aiqclib.common.base.config_base import ConfigBase
from aiqclib.common.base.scikit_learn_model_base import SklearnModelBase


class GaussianNaiveBayes(SklearnModelBase):
    """
    A Gaussian Naive Bayes model wrapper class for training and testing.

    Inherits from :class:`SklearnModelBase` to reuse common Scikit-Learn API logic.

    Features include:

    - Automatic application of ``model_params`` from the YAML config, if defined;
      otherwise, uses default hyperparameters.
    - Uses ``sklearn.naive_bayes.GaussianNB``.

    .. note::
       This class sets :attr:`expected_class_name` to ``"GaussianNaiveBayes"``.
       Naive Bayes does not support the ``n_jobs`` parameter.
    """

    expected_class_name: str = "GaussianNaiveBayes"
    short_name: str = "GNB"

    def __init__(self, config: ConfigBase) -> None:
        """
        Initialize the Gaussian Naive Bayes model with default or user-specified parameters.

        :param config: A configuration object providing model parameters.
        :type config: ConfigBase
        """
        super().__init__(config=config)

        self.model_params: Dict[str, Any] = {
            "priors": None,
            "var_smoothing": 1e-9,
        }
        # Update model parameters with config step parameters
        model_params = self.config.get_model_params(
            self.expected_class_name, self.short_name
        )
        self.model_params.update(model_params)
        self.allow_na = False

    def _get_model_class(self) -> Any:
        """
        Return the Scikit-Learn GaussianNB class.

        :return: The GaussianNB class.
        :rtype: Any
        """
        return SklearnGNB
