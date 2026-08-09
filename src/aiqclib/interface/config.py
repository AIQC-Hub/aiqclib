"""
Module providing utilities for writing YAML configuration templates and
reading them as instantiated configuration objects. Supports "prepare",
"train", "classify", and "nrt_qc" stages using corresponding registry
lookups.
"""

from typing import Optional

from aiqclib.common.base.config_base import ConfigBase
from aiqclib.common.config.classify_config import ClassificationConfig
from aiqclib.common.config.dataset_config import DataSetConfig
from aiqclib.common.config.nrtqc_config import NRTQCConfig
from aiqclib.common.config.training_config import TrainingConfig
from aiqclib.common.config.yaml_templates import (
    get_config_train_set_template,
    get_config_data_set_template,
    get_config_data_set_full_template,
    get_config_data_set_all_template,
    get_config_classify_set_template,
    get_config_classify_set_full_template,
    get_config_nrtqc_template,
)
from aiqclib.common.utils.config import get_config_file
from aiqclib.common.utils.config import read_config as utils_read_config
from aiqclib.common.utils.file import ensure_output_directory


def write_config_template(
    file_name: str,
    stage: str,
    extension: str = "",
    create_dirs: bool = False,
) -> None:
    """
    Write a YAML configuration template for the specified stage
    ("prepare", "train", or "classify") to a file.

    This function:
      1. Chooses a template generator based on the combination of ``stage`` and ``extension``.
      2. Validates that the directory for ``file_name`` exists, creating it when
         ``create_dirs`` is True.
      3. Writes the generated YAML template text to the specified file.

    :param file_name: The path (including filename) where the YAML file will be written.
    :type file_name: str
    :param stage: Determines which template to write; must be one of "prepare",
                  "train", "classify", or "nrt_qc".
    :type stage: str
    :param extension: Determines template extensions; must be one of "", "full", or "reduced".
    :type extension: str
    :param create_dirs: If True, create the output directory (and any missing
                        parents) instead of raising when it does not exist.
                        Defaults to False, so a mistyped path is reported rather
                        than silently creating directories.
    :type create_dirs: bool
    :raises ValueError: If the combined stage and extension is not found in the registry.
    :raises IOError: If the directory of the specified file path does not exist
                     and ``create_dirs`` is False.

    Example Usage:
      >>> # write_config_template("~/new/dir/prepare.yaml", "prepare")
      >>> # IOError: Directory '~/new/dir' does not exist. Create it first, or
      >>> #          pass create_dirs=True to create it automatically. ...
      >>> # write_config_template("/tmp/new/dir/prepare.yaml", "prepare", create_dirs=True)
    """
    function_registry = {
        "prepare_": get_config_data_set_all_template,
        "prepare_full": get_config_data_set_full_template,
        "prepare_reduced": get_config_data_set_template,
        "train_": get_config_train_set_template,
        "classify_": get_config_classify_set_template,
        "classify_full": get_config_classify_set_full_template,
        "nrt_qc_": get_config_nrtqc_template,
    }
    if f"{stage}_{extension}" not in function_registry:
        raise ValueError(f"Module {stage} is not supported.")

    yaml_text = function_registry[f"{stage}_{extension}"]()
    ensure_output_directory(file_name, create_dirs=create_dirs)

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
         (e.g., "data_sets", "training_sets", "classification_sets",
         or "nrt_qc_sets") to map to the corresponding configuration class.
      3. Instantiates and returns the matched configuration class with the resolved path.
      4. If ``set_name`` is provided, it calls the ``select`` method on the instantiated
         configuration object.

    :param file_name: The path (including filename) to the YAML file.
    :type file_name: str
    :param set_name: The name (key) of the desired configuration set within the YAML's dictionary.
                     Defaults to None.
    :type set_name: Optional[str]
    :param auto_select: If True and no ``set_name`` is given, select the file's
                        only set automatically; a file holding several sets
                        raises, since there is nothing to choose between them.
                        Ignored when ``set_name`` names the set to select.
                        Defaults to True.
    :type auto_select: bool
    :return: An instantiated configuration object (:class:`DataSetConfig`,
             :class:`TrainingConfig`, :class:`ClassificationConfig`, or
             :class:`NRTQCConfig`).
    :rtype: ConfigBase
    :raises ValueError: If no valid top-level configuration key is found in the YAML file.
    """
    config_file_name = get_config_file(file_name)
    config = utils_read_config(config_file_name)

    config_classes = {
        "data_sets": DataSetConfig,
        "training_sets": TrainingConfig,
        "classification_sets": ClassificationConfig,
        "nrt_qc_sets": NRTQCConfig,
    }
    matching_key = next((key for key in config_classes.keys() if key in config), None)
    if matching_key is None:
        raise ValueError("No valid 'set' name found in the provided YAML file.")

    # Auto-selection only makes sense when the caller did not name a set: it
    # rejects a file holding several sets, which is precisely the file a caller
    # passing 'set_name' is selecting from.
    config = config_classes[matching_key](
        config_file_name, auto_select and set_name is None
    )

    if set_name is not None:
        config.select(set_name)

    return config
