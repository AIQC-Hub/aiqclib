"""
This module provides utility functions for loading and instantiating various training components,
such as input training sets, model validation classes, and model build classes. It leverages a
registry pattern and a `TrainingConfig` object to determine the specific class to load for each
training step, promoting modularity and configurability in the training pipeline.
"""

from typing import Optional, Dict, Type

import polars as pl

from aiqclib.common.base.dataset_base import DataSetBase
from aiqclib.common.config.training_config import TrainingConfig
from aiqclib.common.loader.training_registry import BUILD_MODEL_REGISTRY
from aiqclib.common.loader.training_registry import INPUT_TRAINING_SET_REGISTRY
from aiqclib.common.loader.training_registry import MODEL_VALIDATION_REGISTRY
from aiqclib.train.step1_read_input.input_base import InputTrainingSetBase
from aiqclib.train.step2_validate_model.validate_base import ValidationBase
from aiqclib.train.step4_build_model.build_model_base import BuildModelBase


def _get_train_class(
    config: TrainingConfig, step: str, registry: Dict[str, Type[DataSetBase]]
) -> Type[DataSetBase]:
    """
    Retrieve the dataset class constructor for a specified training step
    from the provided registry.

    Steps:

      1. Invoke :meth:`TrainingConfig.get_base_class(step)` to get the class name.
      2. Lookup the class in the given ``registry``.
      3. Return the uninstantiated class.

    :param config: A training configuration object, which should provide
                   a base class name for the given step.
    :type config: aiqclib.common.config.training_config.TrainingConfig
    :param step: A string indicating which step name to look up (e.g., "input", "validate", "build").
    :type step: str
    :param registry: A dictionary mapping class names (str) to class types
                     derived from :class:`aiqclib.common.base.dataset_base.DataSetBase`.
    :type registry: Dict[str, Type[aiqclib.common.base.dataset_base.DataSetBase]]
    :raises ValueError: If the retrieved class name is not found in ``registry``.
    :return: The dataset class (uninstantiated) that is used for the specified step.
    :rtype: Type[aiqclib.common.base.dataset_base.DataSetBase]
    """
    class_name = config.get_base_class(step)
    dataset_class = registry.get(class_name)
    if not dataset_class:
        raise ValueError(f"Unknown dataset class specified: {class_name}")

    return dataset_class


def load_step1_input_training_set(config: TrainingConfig) -> InputTrainingSetBase:
    """
    Retrieve and instantiate an :class:`aiqclib.train.step1_read_input.input_base.InputTrainingSetBase` subclass
    for the "input" step, based on the YAML configuration.

    1. Extract the class name with :meth:`TrainingConfig.get_base_class("input")`.
    2. Retrieve the corresponding class from :data:`aiqclib.common.loader.training_registry.INPUT_TRAINING_SET_REGISTRY`.
    3. Instantiate the class and return it.

    :param config: The training configuration object containing a ``base_class``
                   entry under the "input" section.
    :type config: aiqclib.common.config.training_config.TrainingConfig
    :return: An instantiated object of a class that inherits from :class:`aiqclib.train.step1_read_input.input_base.InputTrainingSetBase`.
    :rtype: aiqclib.train.step1_read_input.input_base.InputTrainingSetBase
    """
    dataset_class = _get_train_class(config, "input", INPUT_TRAINING_SET_REGISTRY)
    return dataset_class(config)


def load_step2_model_validation_class(
    config: TrainingConfig, training_sets: Optional[dict[str, pl.DataFrame]] = None
) -> ValidationBase:
    """
    Retrieve and instantiate a :class:`aiqclib.train.step2_validate_model.validate_base.ValidationBase` subclass for
    the "validate" step, based on the YAML configuration.

    Steps:
      1. Extract the class name with :meth:`TrainingConfig.get_base_class("validate")`.
      2. Retrieve the corresponding class from :data:`aiqclib.common.loader.training_registry.MODEL_VALIDATION_REGISTRY`.
      3. Instantiate the class, optionally passing the provided training sets.

    :param config: The training configuration object referencing a ``base_class``
                   under the "validate" section.
    :type config: aiqclib.common.config.training_config.TrainingConfig
    :param training_sets: A dictionary of Polars DataFrames containing data for model validation,
                          defaults to None. Keys typically represent data categories (e.g., "train", "test").
    :type training_sets: Optional[Dict[str, pl.DataFrame]]
    :return: An instantiated object of a class that inherits from :class:`aiqclib.train.step2_validate_model.validate_base.ValidationBase`.
    :rtype: aiqclib.train.step2_validate_model.validate_base.ValidationBase
    """
    dataset_class = _get_train_class(config, "validate", MODEL_VALIDATION_REGISTRY)
    return dataset_class(config, training_sets=training_sets)


def load_step4_build_model_class(
    config: TrainingConfig,
    training_sets: Optional[dict[str, pl.DataFrame]] = None,
    test_sets: Optional[dict[str, pl.DataFrame]] = None,
) -> BuildModelBase:
    """
    Retrieve and instantiate a :class:`aiqclib.train.step4_build_model.build_model_base.BuildModelBase` subclass for
    the "build" step, based on the YAML configuration.

    Steps:
      1. Extract the class name with :meth:`TrainingConfig.get_base_class("build")`.
      2. Retrieve the corresponding class from :data:`aiqclib.common.loader.training_registry.BUILD_MODEL_REGISTRY`.
      3. Instantiate the class, providing any training and test sets.

    :param config: The training configuration object referencing a ``base_class``
                   under the "build" section.
    :type config: aiqclib.common.config.training_config.TrainingConfig
    :param training_sets: A dictionary of Polars DataFrames of training data, defaults to None.
                          Keys typically represent data categories (e.g., "features", "target").
    :type training_sets: Optional[Dict[str, pl.DataFrame]]
    :param test_sets: A dictionary of Polars DataFrames of test data, defaults to None.
                      Keys typically represent data categories (e.g., "features", "target").
    :type test_sets: Optional[Dict[str, pl.DataFrame]]
    :return: An instantiated object of a class that inherits from :class:`aiqclib.train.step4_build_model.build_model_base.BuildModelBase`.
    :rtype: aiqclib.train.step4_build_model.build_model_base.BuildModelBase
    """
    dataset_class = _get_train_class(config, "build", BUILD_MODEL_REGISTRY)
    return dataset_class(
        config,
        training_sets=training_sets,
        test_sets=test_sets,
    )
