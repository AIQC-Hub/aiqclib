"""A set of utilities for handling YAML configuration files.

This module provides utility functions for locating, reading, and parsing
configuration files, typically in YAML format. It facilitates easy retrieval
of specific items within the parsed configuration data.
"""

import os
from typing import Dict, Any, Optional  # Added Optional

import yaml


def get_config_file(config_file: Optional[str]) -> str:
    """Determine the absolute path for a configuration file.

    If the provided path does not exist, a ``FileNotFoundError`` is raised.
    If `config_file` is ``None``, a `ValueError` is raised.

    :param config_file: The path to the configuration file, or ``None``.
    :type config_file: Optional[str]
    :raises ValueError: If `config_file` is ``None``.
    :raises FileNotFoundError: If the path specified by `config_file` does not exist.
    :return: The resolved absolute path to the configuration file.
    :rtype: str
    """
    if config_file is None:
        raise ValueError("Configuration file path cannot be None.")
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"File '{config_file}' does not exist.")

    return str(config_file)


def read_config(config_file: Optional[str]) -> Dict[str, Any]:
    """Read and parse a YAML configuration file.

    This function uses the provided `config_file` path to locate,
    read, and parse a YAML file into a Python dictionary.

    :param config_file: Full path to the config file, or ``None`` to indicate
                        no specific file was provided.
    :type config_file: Optional[str]
    :raises ValueError: If `config_file` is ``None`` (propagated from
                        :func:`get_config_file`).
    :raises FileNotFoundError: If no file is found at the resolved path
                               (propagated from :func:`get_config_file`).
    :raises yaml.YAMLError: If the configuration file is not valid YAML.
    :return: A dictionary representing the parsed YAML configuration.
    :rtype: Dict[str, Any]
    """
    resolved_file = get_config_file(config_file)

    with open(resolved_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data


def get_config_item(config: Dict[str, Any], section: str, name: str) -> Dict[str, Any]:
    """Retrieve a specific item from a section of a configuration dictionary.

    This function iterates through a list of items within a specified
    `section` of the configuration, looking for an item where the ``"name"``
    key matches the given `name`.

    :param config: The configuration dictionary, e.g., from :func:`read_config`.
    :type config: Dict[str, Any]
    :param section: The top-level key in `config` that contains a list of items.
    :type section: str
    :param name: The value of the "name" key to match within the item.
    :type name: str
    :raises KeyError: If the `section` does not exist in the `config` dictionary.
    :raises TypeError: If the value at `config[section]` is not iterable.
    :raises ValueError: If no item with the specified `name` is found in the section.
    :return: The dictionary of the matching configuration item.
    :rtype: Dict[str, Any]
    """
    for item in config[section]:
        if item.get("name") == name:
            return item

    raise ValueError(f"Item with name '{name}' not found in section '{section}'.")
