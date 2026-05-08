"""
Module for handling YAML-based configuration management.

This module provides the `ConfigBase` abstract base class, which facilitates
loading, validating, and retrieving structured data from YAML configuration files.
It uses JSON schemas for validation and supports template-based configuration loading.
"""

import os
from abc import ABC
from typing import List, Dict, Optional

import jsonschema
import yaml
from jsonschema import validate

from aiqclib.common.config.yaml_schema import (
    get_data_set_config_schema,
    get_training_config_schema,
    get_classification_config_schema,
)
from aiqclib.common.config.yaml_templates import (
    get_config_data_set_template,
    get_config_data_set_full_template,
    get_config_train_set_template,
    get_config_classify_set_template,
    get_config_classify_set_full_template,
)
from aiqclib.common.utils.config import get_config_item
from aiqclib.common.utils.config import read_config


class ConfigBase(ABC):
    """
    Abstract base class for loading and accessing YAML configurations.

    This class provides a common interface for handling configuration files.
    It supports loading from a file path or from a built-in template,
    validating the configuration against a predefined JSON schema, and
    providing convenient methods to access specific parts of the config.

    Subclasses must override the ``expected_class_name`` attribute to match
    the ``base_class`` value specified in the YAML configuration.

    .. note::
       This is an abstract base class and should not be instantiated directly.

    :ivar expected_class_name: Must be overridden by subclasses to match the
                               YAML's ``base_class`` entry.
    :vartype expected_class_name: str, optional
    :ivar section_name: The top-level section of the config this instance manages.
    :vartype section_name: str
    :ivar yaml_schema: The JSON schema used for validating the configuration.
    :vartype yaml_schema: dict
    :ivar full_config: The entire configuration loaded from the YAML file.
    :vartype full_config: dict
    :ivar valid_yaml: flag indicating if the loaded configuration is valid.
    :vartype valid_yaml: bool
    :ivar data: The specific configuration dictionary for the selected entry.
    :vartype data: dict, optional
    :ivar dataset_name: The name of the selected dataset or task.
    :vartype dataset_name: str, optional
    """

    expected_class_name = None  # Must be overridden by child classes

    def __init__(
        self, section_name: str, config_file: str, auto_select: bool = False
    ) -> None:
        """
        Initialize the configuration object from a YAML file or template.

        :param section_name: The name of the configuration section to load.
        :type section_name: str
        :param config_file: Path to the YAML file or a template identifier.
        :type config_file: str
        :param auto_select: If True, automatically selects the entry if only one exists.
        :type auto_select: bool
        :raises NotImplementedError: If ``expected_class_name`` is not defined.
        :raises ValueError: If the section name or template name is unsupported.
        """
        if not self.expected_class_name:
            raise NotImplementedError(
                "Child class must define 'expected_class_name' attribute"
            )

        yaml_schemas = {
            "data_sets": get_data_set_config_schema,
            "data_sets_with_norm": get_data_set_config_schema,
            "training_sets": get_training_config_schema,
            "classification_sets": get_classification_config_schema,
            "classification_sets_with_norm": get_classification_config_schema,
        }
        if section_name not in yaml_schemas:
            raise ValueError(f"Section name {section_name} is not supported.")

        yaml_templates = {
            "template:data_sets": get_config_data_set_template,
            "template:data_sets_full": get_config_data_set_full_template,
            "template:training_sets": get_config_train_set_template,
            "template:classification_sets": get_config_classify_set_template,
            "template:classification_sets_full": get_config_classify_set_full_template,
        }
        if str(config_file).startswith("template:"):
            if str(config_file) not in yaml_templates:
                raise ValueError(f"Template name {config_file} is not supported.")
            full_config = yaml.safe_load(yaml_templates.get(str(config_file))())
        else:
            full_config = read_config(config_file)

        self.section_name: str = section_name
        self.yaml_schema: Dict = yaml.safe_load(yaml_schemas.get(section_name)())
        self.full_config: Dict = full_config
        self.valid_yaml: bool = False
        self.data: Optional[Dict] = None
        self.dataset_name: Optional[str] = None

        if auto_select:
            self.auto_select()

    def auto_select(self) -> None:
        """
        Automatically validate and select a single configuration entry.

        :raises ValueError: If the YAML is invalid or multiple entries exist.
        :return: None
        :rtype: NoneType
        """
        message = self.validate()
        if not self.valid_yaml:
            raise ValueError(message)

        if len(self.full_config[self.section_name]) == 1:
            self.select(self.full_config[self.section_name][0]["name"])
        else:
            raise ValueError(
                "'auto_select' option is invalid when there are multiple data set names"
            )

    def validate(self) -> str:
        """
        Validate the loaded configuration against the corresponding schema.

        :return: A message indicating whether validation succeeded or failed.
        :rtype: str
        """
        try:
            validate(instance=self.full_config, schema=self.yaml_schema)
            self.valid_yaml = True
            return "YAML file is valid"
        except jsonschema.exceptions.ValidationError as e:
            self.valid_yaml = False
            return f"YAML file is invalid: {e.message}"

    def select(self, dataset_name: str) -> None:
        """
        Select and load a specific configuration entry from the YAML.

        :param dataset_name: The name of the configuration to select.
        :type dataset_name: str
        :raises ValueError: If validation fails or the dataset name is not found.
        :return: None
        :rtype: NoneType
        """
        message = self.validate()
        if not self.valid_yaml:
            raise ValueError(message)

        self.data = get_config_item(
            self.full_config, self.section_name, dataset_name
        ).copy()
        self.data["path_info"] = get_config_item(
            self.full_config, "path_info_sets", self.data["path_info"]
        )
        self.dataset_name = dataset_name

    def get_base_path(self, step_name: str) -> str:
        """
        Retrieve the base path for a given processing step.

        :param step_name: The name of the step (e.g., "preprocess").
        :type step_name: str
        :return: The configured base path.
        :rtype: str
        :raises ValueError: If no base path is found.
        """
        if step_name not in self.data["path_info"] or (
            step_name in self.data["path_info"]
            and "base_path" not in self.data["path_info"][step_name]
        ):
            step_name = "common"
        base_path = self.data["path_info"][step_name].get("base_path")

        if base_path is None:
            raise ValueError(
                f"'base_path' for '{step_name}' not found or set to None in the config file"
            )

        return base_path

    def get_summary_stats(self, stats_name: str, stats_type: str = "min_max") -> Dict:
        """
        Retrieve specific summary statistics parameters from the configuration.

        :param stats_name: Name of the summary statistics set to retrieve.
        :type stats_name: str
        :param stats_type: Type of statistics (e.g., "min_max"). Defaults to "min_max".
        :type stats_type: str
        :raises ValueError: If the specified stats name is not found.
        :return: A dictionary containing the requested statistics.
        :rtype: dict
        """
        for d in self.data["feature_stats_set"].get(stats_type, []):
            if d["name"] == stats_name:
                return d["stats"]

        raise ValueError(
            f"Summary statistics set '{stats_name}' not found in the config file."
        )

    def get_step_params(self, step_name: str) -> Dict:
        """
        Retrieve the parameters dictionary for a specific step.

        :param step_name: The name of the step.
        :type step_name: str
        :return: Parameters for the specified step.
        :rtype: dict
        :raises KeyError: If the step or param set is missing.
        """
        return self.data["step_param_set"]["steps"][step_name]

    def get_model_params(self, model_long_name: str, model_short_name: str) -> Dict:
        """
        Retrieve the parameters dictionary for a model.

        :param model_long_name: The long-form name of the model.
        :type model_long_name: str
        :param model_short_name: The short-form name of the model.
        :type model_short_name: str
        :return: Parameters for the specified model or the whole model param dict.
        :rtype: dict
        """
        model_params = self.data["step_param_set"]["steps"]["model"].get(
            "model_params", {}
        )

        if model_long_name in model_params:
            return model_params[model_long_name]
        elif model_short_name in model_params:
            return model_params[model_short_name]
        else:
            return model_params

    def get_dataset_folder_name(self, step_name: str) -> str:
        """
        Get the dataset-specific folder name for a given step.

        :param step_name: The name of the step.
        :type step_name: str
        :return: The folder name for the dataset, or an empty string.
        :rtype: str
        """
        dataset_folder_name = self.data.get("dataset_folder_name", "")

        if (
            step_name in self.data["step_param_set"]["steps"]
            and "dataset_folder_name" in self.data["step_param_set"]["steps"][step_name]
        ):
            dataset_folder_name = self.get_step_params(step_name).get(
                "dataset_folder_name", ""
            )

        return dataset_folder_name

    def get_step_folder_name(
        self, step_name: str, folder_name_auto: bool = True
    ) -> str:
        """
        Get the folder name for a specific processing step.

        :param step_name: The name of the step.
        :type step_name: str
        :param folder_name_auto: If True, uses step_name as fallback. Defaults to True.
        :type folder_name_auto: bool
        :return: The folder name for the step.
        :rtype: str
        """
        orig_step_name = step_name
        if step_name not in self.data["path_info"] or (
            step_name in self.data["path_info"]
            and "step_folder_name" not in self.data["path_info"][step_name]
        ):
            step_name = "common"
        step_folder_name = self.data["path_info"][step_name].get("step_folder_name")

        if step_folder_name is None:
            step_folder_name = orig_step_name if folder_name_auto else ""

        return step_folder_name

    def get_file_name(self, step_name: str, default_name: Optional[str] = None) -> str:
        """
        Retrieve the file name for a given step.

        :param step_name: The name of the step.
        :type step_name: str
        :param default_name: Fallback file name if not defined in config.
        :type default_name: str, optional
        :return: The file name for the step.
        :rtype: str
        :raises ValueError: If no file name is found and no default is provided.
        """
        file_name = default_name
        if (
            step_name in self.data["step_param_set"]["steps"]
            and "file_name" in self.data["step_param_set"]["steps"][step_name]
        ):
            file_name = self.data["step_param_set"]["steps"][step_name].get(
                "file_name", ""
            )

        if file_name is None:
            raise ValueError(
                f"'file_name' for '{step_name}' not found or set to None in the config file"
            )

        return file_name

    def get_full_file_name(
        self,
        step_name: str,
        default_file_name: Optional[str] = None,
        use_dataset_folder: bool = True,
        folder_name_auto: bool = True,
    ) -> str:
        """
        Construct a full, normalized file path for a step.

        :param step_name: The name of the step.
        :type step_name: str
        :param default_file_name: Default file name if not in config.
        :type default_file_name: str, optional
        :param use_dataset_folder: If True, include dataset folder. Defaults to True.
        :type use_dataset_folder: bool
        :param folder_name_auto: If True, auto-generate step folder name. Defaults to True.
        :type folder_name_auto: bool
        :return: The complete, normalized file path.
        :rtype: str
        """
        base_path = self.get_base_path(step_name)
        dataset_folder_name = (
            self.get_dataset_folder_name(step_name) if use_dataset_folder else ""
        )
        folder_name = self.get_step_folder_name(step_name, folder_name_auto)
        file_name = self.get_file_name(step_name, default_file_name)

        return os.path.normpath(
            os.path.join(base_path, dataset_folder_name, folder_name, file_name)
        )

    def get_base_class(self, step_name: str) -> str:
        """
        Retrieve the associated class name for a specified step.

        :param step_name: The name of the step.
        :type step_name: str
        :return: The class name defined for the step.
        :rtype: str
        """
        return self.data["step_class_set"]["steps"][step_name]

    def set_base_class(self, step_name: str, value: str) -> None:
        """
        Set the associated class name for a specified step.

        :param step_name: The name of the step.
        :type step_name: str
        :param value: The class name value to set.
        :type value: str
        :return: None
        :rtype: NoneType
        """
        self.data["step_class_set"]["steps"][step_name] = value

    def get_target_variables(self) -> List[Dict]:
        """
        Get the list of target variable definitions from the configuration.

        :return: List of target variable definition dictionaries.
        :rtype: list[dict]
        """
        return self.data["target_set"]["variables"]

    def get_target_names(self) -> List[str]:
        """
        Get the names of all target variables.

        :return: List of target variable names.
        :rtype: list[str]
        """
        return [x["name"] for x in self.get_target_variables()]

    def get_target_dict(self) -> Dict[str, Dict]:
        """
        Get target variable definitions as a name-keyed dictionary.

        :return: Mapping of target names to their definitions.
        :rtype: dict[str, dict]
        """
        return {x["name"]: x for x in self.get_target_variables()}

    def get_target_file_names(
        self,
        step_name: str,
        default_file_name: Optional[str] = None,
        use_dataset_folder: bool = True,
        folder_name_auto: bool = True,
    ) -> Dict[str, str]:
        """
        Construct a dictionary of full file paths for each target variable.

        :param step_name: The name of the step.
        :type step_name: str
        :param default_file_name: Default file name template.
        :type default_file_name: str, optional
        :param use_dataset_folder: If True, include dataset folder. Defaults to True.
        :type use_dataset_folder: bool
        :param folder_name_auto: If True, auto-generate step folder name. Defaults to True.
        :type folder_name_auto: bool
        :return: Dictionary mapping target names to formatted file paths.
        :rtype: dict[str, str]
        """
        full_file_name = self.get_full_file_name(
            step_name, default_file_name, use_dataset_folder, folder_name_auto
        )
        return {
            x: full_file_name.replace("{target_name}", x)
            for x in self.get_target_names()
        }

    def update_feature_param_with_stats(self) -> None:
        """
        Update feature parameters with corresponding summary statistics in-place.

        :return: None
        :rtype: NoneType
        """
        for x in self.data["feature_param_set"]["params"]:
            if ("stats_set" in x) and (x["stats_set"]["type"] != "raw"):
                x["stats"] = self.get_summary_stats(
                    x["stats_set"]["name"], x["stats_set"]["type"]
                )

    def __repr__(self) -> str:
        """
        Return a string representation of the configuration object.

        :return: String identifying the instance and its managed section.
        :rtype: str
        """
        return f"ConfigBase(section_name={self.section_name})"
