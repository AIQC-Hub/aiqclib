"""
This module provides utility functions for loading and managing model classes
based on configuration settings, typically used in a machine learning or data
processing pipeline.
"""

from typing import Optional, Type

from aiqclib.common.base.config_base import ConfigBase
from aiqclib.common.base.model_base import ModelBase
from aiqclib.common.loader.single_model_registry import SINGLE_MODEL_REGISTRY


def load_single_model_class(config: ConfigBase) -> ModelBase:
    """
    Retrieve and instantiate a model class for the "model" step from the provided configuration.

    This function performs the following steps:

    1. Fetches the class name from the configuration using ``config.get_base_class("model")``.
    2. Looks up the corresponding class in the global :data:`~aiqclib.common.loader.single_model_registry.SINGLE_MODEL_REGISTRY`.
    3. Instantiates the found class with the given configuration object as an argument.

    :param config: A configuration object that includes a "base_class" entry
                   under the "model" step, specifying which model class to load.
                   This object must implement the ``get_base_class`` method.
    :type config: aiqclib.common.base.config_base.ConfigBase
    :returns: An instantiated model object, which is an instance of a class
              inheriting from :class:`~aiqclib.common.base.model_base.ModelBase`.
    :rtype: aiqclib.common.base.model_base.ModelBase
    :raises ValueError: If the retrieved model class name is not found in
                        the :data:`~aiqclib.common.loader.single_model_registry.SINGLE_MODEL_REGISTRY`.
    """
    class_name: str = config.get_base_class("model")
    return load_single_model_class_with_class_name(config, class_name)


def load_single_model_class_with_class_name(
    config: ConfigBase, class_name: str
) -> ModelBase:
    """
    Retrieves and instantiates a specific model class using a given configuration and class name.

    This function looks up the specified model class name in the global
    :data:`~aiqclib.common.loader.single_model_registry.SINGLE_MODEL_REGISTRY`
    and then instantiates it with the provided configuration object.

    :param config: The configuration object to be passed to the model class constructor.
                   This object must implement the ``get_base_class`` method if it were
                   to be used for fetching the class name, but here it's directly passed
                   to the model constructor.
    :type config: aiqclib.common.base.config_base.ConfigBase
    :param class_name: The string name of the model class to retrieve and instantiate.
                       This name must exist as a key in the
                       :data:`~aiqclib.common.loader.single_model_registry.SINGLE_MODEL_REGISTRY`.
    :type class_name: str
    :returns: An instantiated model object, which is an instance of a class
              inheriting from :class:`~aiqclib.common.base.model_base.ModelBase`.
    :rtype: aiqclib.common.base.model_base.ModelBase
    :raises ValueError: If the provided ``class_name`` is not found in
                        the :data:`~aiqclib.common.loader.single_model_registry.SINGLE_MODEL_REGISTRY`.
    """
    model_class: Optional[Type[ModelBase]] = SINGLE_MODEL_REGISTRY.get(class_name)
    if not model_class:
        raise ValueError(f"Unknown model class specified: {class_name}")

    return model_class(config)
