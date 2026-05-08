"""
Module providing utilities for writing YAML configuration templates and
reading them as instantiated configuration objects. Supports "prepare",
"train", and "classify" stages using corresponding registry lookups.
"""

import os
from typing import Optional

from aiqclib.common.base.config_base import ConfigBase
from aiqclib.common.config.classify_config import ClassificationConfig
from aiqclib.common.config.dataset_config import DataSetConfig
from aiqclib.common.config.training_config import TrainingConfig
from aiqclib.common.config.yaml_templates import (
    get_config_train_set_template,
    get_config_data_set_template,
    get_config_data_set_full_template,
    get_config_data_set_all_template,
    get_config_classify_set_template,
    get_config_classify_set_full_template,
)
from aiqclib.common.utils.config import get_config_file
from aiqclib.common.utils.config import read_config as utils_read_config


def write_config_template(file_name: str, stage: str, extension: str = "") -> None:
    """
    Write a YAML configuration template for the specified stage
    ("prepare", "train", or "classify") to a file.

    This function:
      1. Chooses a template generator based on the combination of ``stage`` and ``extension``.
      2. Validates that the directory for ``file_name`` exists.
      3. Writes the generated YAML template text to the specified file.

    :param file_name: The path (including filename) where the YAML file will be written.
    :type file_name: str
    :param stage: Determines which template to write; must be one of "prepare", "train", or "classify".
    :type stage: str
    :param extension: Determines template extensions; must be one of "", "full", or "reduced".
    :type extension: str
    :raises ValueError: If the combined stage and extension is not found in the registry.
    :raises IOError: If the directory of the specified file path does not exist.
    """
    function_registry = {
        "prepare_": get_config_data_set_all_template,
        "prepare_full": get_config_data_set_full_template,
        "prepare_reduced": get_config_data_set_template,
        "train_": get_config_train_set_template,
        "classify_": get_config_classify_set_template,
        "classify_full": get_config_classify_set_full_template,
    }
    if f"{stage}_{extension}" not in function_registry:
        raise ValueError(f"Module {stage} is not supported.")

    yaml_text = function_registry[f"{stage}_{extension}"]()
    dir_path = os.path.dirname(file_name)
    if not os.path.exists(dir_path) and dir_path != "":
        raise IOError(f"Directory '{dir_path}' does not exist.")

    with open(file_name, "w", encoding="utf-8") as yaml_file:
        yaml_file.write(yaml_text)


def read_config(
    file_name: str, set_name: Optional[str] = None, auto_select: bool = True
) -> ConfigBase:
    """
    Read a YAML configuration file as a :class:`ConfigBase` object,
    automatically selecting the appropriate subclass based on the content.

    This function:
      1. Resolves the file path by calling :func:`aiqclib.common.utils.config.get_config_file`.
      2. Reads the specified YAML file and identifies the main key
         (e.g., "data_sets", "training_sets", or "classification_sets")
         to map to the corresponding configuration class.
      3. Instantiates and returns the matched configuration class with the resolved path.
      4. If ``set_name`` is provided, it calls the ``select`` method on the instantiated
         configuration object.

    :param file_name: The path (including filename) to the YAML file.
    :type file_name: str
    :param set_name: The name (key) of the desired configuration set within the YAML's dictionary.
                     Defaults to None.
    :type set_name: Optional[str]
    :param auto_select: If True, the first available data set name will be selected automatically
                        if no specific ``set_name`` is provided. Defaults to True.
    :type auto_select: bool
    :return: An instantiated configuration object (either :class:`DataSetConfig`,
             :class:`TrainingConfig`, or :class:`ClassificationConfig`).
    :rtype: ConfigBase
    :raises ValueError: If no valid top-level configuration key is found in the YAML file.
    """
    config_file_name = get_config_file(file_name)
    config = utils_read_config(config_file_name)

    config_classes = {
        "data_sets": DataSetConfig,
        "training_sets": TrainingConfig,
        "classification_sets": ClassificationConfig,
    }
    matching_key = next((key for key in config_classes.keys() if key in config), None)
    if matching_key is None:
        raise ValueError("No valid 'set' name found in the provided YAML file.")

    config = config_classes[matching_key](config_file_name, auto_select)

    if set_name is not None:
        config.select(set_name)

    return config
