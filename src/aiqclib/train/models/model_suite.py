"""
This module provides the :class:`ModelSuite` class, inheriting
from :class:`aiqclib.common.base.model_base.ModelBase`.

It facilitates training, prediction, and evaluation with multiple ML methods.
"""

import copy

from typing import Dict, Any, List, Self

from aiqclib.common.base.config_base import ConfigBase
from aiqclib.common.base.model_base import ModelBase

from aiqclib.common.loader.single_model_loader import (
    load_single_model_class_with_class_name,
)


class ModelSuite(ModelBase):
    """
    A model suite class for training and testing multiple machine learning models.

    Inherits from :class:`aiqclib.common.base.model_base.ModelBase` to use the
    common model class interface.

    Features include:

    - Automatic application of ``model_params`` from the YAML configuration, if defined.
    - Manages a collection of different machine learning models for comparative analysis
      or ensemble operations.

    .. note::
       This class sets :attr:`expected_class_name` to ``"ModelSuite"``.
    """

    expected_class_name: str = "ModelSuite"
    short_name: str = "MS"
    multi = True

    def __init__(self, config: ConfigBase) -> None:
        """
        Initialize the ModelSuite with a configuration object.

        This constructor sets up the collection of machine learning models
        (methods) based on the provided configuration or a set of default methods.
        Each method is then loaded and instantiated.

        :param config: A configuration object providing model parameters and
                       defining which ML methods to include in the suite.
        :type config: :class:`aiqclib.common.base.config_base.ConfigBase`
        """
        super().__init__(config=config)

        self.default_methods: List[str] = [
            "Logit",
            "LDA",
            "SVM",
            "DT",
            "XGB",
            "RF",
            "GNB",
            "KNN",
            "MLP",
        ]
        self.methods = self.config.get_step_params("model").get(
            "methods", self.default_methods
        )

        self.method_objs: Dict[str, Any] = {
            m: self._load_model_class_with_method_name(config, m) for m in self.methods
        }

    def _load_model_class_with_method_name(
        self, config: ConfigBase, method: str
    ) -> Any:
        """
        Loads and instantiates a single model class based on its method name.

        This private helper method creates a deep copy of the configuration,
        sets the base class for the model step to the specified method, and
        then uses a utility function to load the model class.

        :param config: The base configuration object. A deep copy of this
                       will be modified to specify the current method.
        :type config: :class:`aiqclib.common.base.config_base.ConfigBase`
        :param method: The name of the machine learning method (e.g., "Logit", "SVM").
        :type method: str
        :returns: An instantiated model object corresponding to the specified method.
        :rtype: Any
        """
        config_method = copy.deepcopy(config)
        config_method.set_base_class("model", method)
        return load_single_model_class_with_class_name(config_method, method)

    def _get_model_class(self) -> Any:
        """
        Retrieves the model class.

        In the context of ModelSuite, which manages multiple models, this method
        is overridden from :class:`aiqclib.common.base.model_base.ModelBase` but
        does not return a single model class directly. It is typically not used
        for ModelSuite itself.

        :returns: None, as ModelSuite manages multiple distinct model instances,
                  rather than providing a single class here.
        :rtype: None
        """
        return None  # pragma: no cover

    def set_enable_shap(self, enable_shap: bool):
        """
        Sets the SHAP explanation flag for all models within the suite.

        Iterates through all instantiated model objects (``method_objs``)
        and sets their ``enable_shap`` attribute to the specified value.
        This allows for global control over SHAP explanation generation across models.

        :param enable_shap: A boolean flag to enable or disable SHAP explanations.
        :type enable_shap: bool
        """
        for method_obj in self.method_objs.values():
            method_obj.enable_shap = enable_shap

    def build(self) -> None:
        """
        Builds the model architecture or pipeline for all models in the suite.

        This method is intended to orchestrate the build process for each
        individual model managed by the ModelSuite. Subclasses or implementations
        could iterate through `self.method_objs` and call a `build` method
        on each if available, or perform other suite-level setup.
        As implemented, it currently does nothing (pass).
        """
        pass  # pragma: no cover

    def test(self) -> None:
        """
        Evaluates the performance of the models in the suite on a test set.

        This method is designed to coordinate the testing phase for each
        individual model. Implementations might iterate through `self.method_objs`,
        call a `test` method on each, and aggregate results, or perform
        suite-level validation.
        As implemented, it currently does nothing (pass).
        """
        pass  # pragma: no cover

    def update_nthreads(self, model: Self) -> Self:
        """
        Updates the number of threads for the given ModelSuite instance and its internal models.

        This method is intended to update the thread configuration for all
        sub-models managed within the provided `ModelSuite` instance.
        The current implementation (`pass`) indicates it does not perform
        any operations. It expects to receive an instance of `ModelSuite`
        and is designed to return the modified instance.

        :param model: The ModelSuite instance whose internal models' thread
                      configurations need to be updated.
        :type model: :class:`ModelSuite`
        :returns: The updated ModelSuite instance, potentially with modified
                  thread settings for its constituent models.
        :rtype: :class:`ModelSuite`

        .. warning::
           The current implementation of this method does nothing (`pass`).
           Its intended functionality regarding how it interacts with the
           `model` parameter (if it's `self` or another instance) and how
           it updates thread counts for individual sub-models is unclear
           and needs further implementation.
        """
        pass  # pragma: no cover
