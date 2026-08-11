"""
Module for handling YAML-based configuration management.

This module provides the `ConfigBase` abstract base class, which facilitates
loading, validating, and retrieving structured data from YAML configuration files.
It uses JSON schemas for validation and supports template-based configuration loading.
"""

import os
import textwrap
from abc import ABC
from typing import List, Dict, Optional, Tuple

import jsonschema
import yaml
from jsonschema import validate

from aiqclib.common.config.yaml_schema import (
    get_data_set_config_schema,
    get_training_config_schema,
    get_classification_config_schema,
    get_nrtqc_config_schema,
)
from aiqclib.common.config.yaml_templates import (
    get_config_data_set_template,
    get_config_data_set_full_template,
    get_config_train_set_template,
    get_config_classify_set_template,
    get_config_classify_set_full_template,
    get_config_nrtqc_template,
)
from aiqclib.common.utils.config import get_config_item
from aiqclib.common.utils.config import read_config
from aiqclib.common.utils.file import expand_path


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
    :ivar config_file: The YAML path or ``template:`` identifier this instance
                       was loaded from, kept for reporting.
    :vartype config_file: str
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
            "nrt_qc_sets": get_nrtqc_config_schema,
        }
        if section_name not in yaml_schemas:
            raise ValueError(f"Section name {section_name} is not supported.")

        yaml_templates = {
            "template:data_sets": get_config_data_set_template,
            "template:data_sets_full": get_config_data_set_full_template,
            "template:training_sets": get_config_train_set_template,
            "template:classification_sets": get_config_classify_set_template,
            "template:classification_sets_full": get_config_classify_set_full_template,
            "template:nrt_qc_sets": get_config_nrtqc_template,
        }
        if str(config_file).startswith("template:"):
            if str(config_file) not in yaml_templates:
                raise ValueError(f"Template name {config_file} is not supported.")
            full_config = yaml.safe_load(yaml_templates.get(str(config_file))())
        else:
            full_config = read_config(config_file)

        self.section_name: str = section_name
        self.config_file: str = str(config_file)
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

    def check_schema(self) -> Tuple[bool, str]:
        """
        Check the loaded configuration against the schema without storing
        the outcome.

        This is the side-effect-free half of :meth:`validate`, so that
        reporting code (:meth:`summary`) can describe the configuration
        without changing :attr:`valid_yaml` underneath a caller.

        :return: Whether the configuration is valid, and a message describing
                 the outcome.
        :rtype: tuple[bool, str]
        """
        try:
            validate(instance=self.full_config, schema=self.yaml_schema)
            return True, "YAML file is valid"
        except jsonschema.exceptions.ValidationError as e:
            return False, f"YAML file is invalid: {e.message}"

    def validate(self) -> str:
        """
        Validate the loaded configuration against the corresponding schema,
        storing the outcome in :attr:`valid_yaml`.

        :return: A message indicating whether validation succeeded or failed.
        :rtype: str
        """
        self.valid_yaml, message = self.check_schema()
        return message

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

        A leading ``~`` is expanded here, so ``base_path: ~/aiqc_project/data``
        in a configuration file means the same directory it would mean in a
        shell rather than a literal ``~`` folder under the working directory.

        :param step_name: The name of the step (e.g., "preprocess").
        :type step_name: str
        :return: The configured base path, with ``~`` expanded.
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

        return expand_path(base_path)

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

    @staticmethod
    def is_flag_missing(target_value: Dict) -> bool:
        """
        Return True when a target variable has no usable QC ``flag`` defined.

        A flag is considered missing when the ``flag`` key is absent, ``None``,
        or an empty/whitespace-only string. This is the trigger for the
        label-free (``skip_evaluation``) classification path.

        :param target_value: A single target variable definition.
        :type target_value: dict
        :return: True if no usable flag column is specified, else False.
        :rtype: bool
        """
        flag = target_value.get("flag")
        return flag is None or (isinstance(flag, str) and flag.strip() == "")

    def get_skip_evaluation(self, target_name: str) -> bool:
        """
        Resolve whether performance evaluation and label creation should be
        skipped for a given classification target.

        Resolution order:

          1. If ``skip_evaluation`` is explicitly set in the ``model`` step
             params, that value wins for every target in the step.
          2. Otherwise it is derived per target: True when the target's QC
             ``flag`` is missing/empty (see :meth:`is_flag_missing`).

        :param target_name: The name of the target variable.
        :type target_name: str
        :return: True to skip label creation and performance evaluation.
        :rtype: bool
        """
        override = self.get_step_params("model").get("skip_evaluation")
        if override is not None:
            return bool(override)

        target_value = self.get_target_dict().get(target_name, {})
        return self.is_flag_missing(target_value)

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

    def update_feature_param_with_stats(
        self, types: Optional[List[str]] = None
    ) -> None:
        """
        Update feature parameters with corresponding summary statistics in-place.

        For each feature whose ``stats_set.type`` is a scaling type (i.e. not
        ``raw``), the resolved statistics are looked up in
        :attr:`data`'s ``feature_stats_set`` (by name and type) and stored under
        the feature's ``stats`` key, ready for use by the feature classes.

        :param types: If provided, only resolve features whose ``stats_set.type``
                      is in this list. This allows the manually-supplied
                      ``min_max`` statistics to be resolved at configuration-load
                      time while deferring the data-derived ``auto_min_max`` and
                      ``standard`` statistics until after the summary statistics
                      have been computed. If ``None``, every non-``raw`` feature
                      is resolved (the historical behaviour).
        :type types: Optional[List[str]]
        :return: None
        :rtype: NoneType
        """
        for x in self.data["feature_param_set"]["params"]:
            if "stats_set" not in x:
                continue
            stats_type = x["stats_set"]["type"]
            if stats_type == "raw":
                continue
            if types is not None and stats_type not in types:
                continue
            stats_name = x["stats_set"].get("name", x.get("feature"))
            x["stats"] = self.get_summary_stats(stats_name, stats_type)

    def get_normalization_file_name(
        self, default_file_name: str = "normalization_stats.yaml"
    ) -> str:
        """
        Resolve the full path of the normalization statistics file.

        This file holds the data-derived normalization values (for
        ``auto_min_max`` and ``standard`` features). It is written during
        dataset preparation and read back during classification so that the
        identical fitted normalization is applied without re-entering values.

        The path is resolved through the standard step-path machinery using the
        logical step name ``"normalize"``. The folder defaults to ``normalize``
        and the file name can be overridden via
        ``step_param_sets.steps.normalize.file_name`` in the configuration.

        :param default_file_name: File name used when none is set in the config.
        :type default_file_name: str
        :return: The complete, normalized path to the normalization file.
        :rtype: str
        """
        return self.get_full_file_name(
            step_name="normalize", default_file_name=default_file_name
        )

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    #: Width of the label column used by :meth:`summary`, including its
    #: two-space indent. Continuation lines are indented to match.
    _label_width: int = 12

    #: Steps whose paths are resolved without the dataset folder, i.e. those
    #: the step classes read with ``use_dataset_folder=False``. Only
    #: :meth:`summary` uses this — the step classes remain the authority on
    #: their own paths, so a subclass listing the wrong steps misreports a
    #: directory rather than changing where anything is written.
    _steps_without_dataset_folder: Tuple[str, ...] = ("input",)

    @classmethod
    def _wrap(cls, text: str, width: int) -> List[str]:
        """
        Wrap a comma-separated value to the summary's content column.

        :param text: The text to wrap.
        :type text: str
        :param width: The total line width the summary is formatted to.
        :type width: int
        :return: The wrapped lines, without indentation.
        :rtype: list[str]
        """
        return textwrap.wrap(text, width=max(width - cls._label_width, 20)) or [""]

    def _entry_names(self) -> List[str]:
        """
        List the names of every entry in this instance's section.

        :return: The entry names, or an empty list if the section is missing
                 or malformed.
        :rtype: list[str]
        """
        try:
            return [x["name"] for x in self.full_config[self.section_name]]
        except (KeyError, TypeError, IndexError):
            return []

    def _summary_targets(self) -> List[str]:
        """
        Describe each target variable and the flag values it labels from.

        :return: One line per target, or an empty list when no target set is
                 resolved.
        :rtype: list[str]
        """
        try:
            variables = self.get_target_variables()
        except (KeyError, TypeError):
            return []

        lines = []
        for variable in variables:
            if self.is_flag_missing(variable):
                detail = "no flag"
            else:
                detail = f"flag {variable['flag']}"
                for key, label in (
                    ("pos_flag_values", "pos"),
                    ("neg_flag_values", "neg"),
                ):
                    if variable.get(key) is not None:
                        detail += f", {label} {variable[key]}"
            lines.append(f"{variable.get('name', '?')} ({detail})")
        return lines

    def _summary_input_file(self) -> List[str]:
        """
        Resolve the input file the pipeline will read for this configuration.

        :return: A single-element list holding the resolved path, or an empty
                 list when the configuration names no input file.
        :rtype: list[str]
        """
        file_name = (self.data or {}).get("input_file_name")
        if not file_name:
            return []
        try:
            return [
                self.get_full_file_name(
                    "input",
                    default_file_name=file_name,
                    use_dataset_folder="input"
                    not in self._steps_without_dataset_folder,
                )
            ]
        except (KeyError, TypeError, ValueError):
            return [file_name]

    def _summary_filters(self, width: int) -> List[str]:
        """
        Describe the active row filters of the ``input`` step.

        These are worth surfacing because a filter that matches nothing — a
        ``keep_years`` naming years the input does not cover, say — empties
        the dataset without any hint in the configuration itself.

        :param width: The total line width the summary is formatted to.
        :type width: int
        :return: The wrapped filter description, or an empty list when no
                 filter is set.
        :rtype: list[str]
        """
        try:
            params = self.get_step_params("input")
        except (KeyError, TypeError):
            return []

        active = [
            f"{key} {value}"
            for key, value in (params.get("filter_method_dict") or {}).items()
            if value
        ]
        if not active:
            return []

        text = ", ".join(active)
        if (params.get("sub_steps") or {}).get("filter_rows") is False:
            text += "  (not applied: sub_steps.filter_rows is false)"
        return self._wrap(text, width)

    def _step_directory(self, step_name: str) -> str:
        """
        Compose the output directory a step writes into.

        This is :meth:`get_full_file_name` without the file name, which the
        step classes supply themselves and which is therefore not knowable
        from the configuration alone.

        :param step_name: The name of the step.
        :type step_name: str
        :return: The directory, or a placeholder when it cannot be resolved.
        :rtype: str
        """
        try:
            dataset_folder = (
                ""
                if step_name in self._steps_without_dataset_folder
                else self.get_dataset_folder_name(step_name)
            )
            return os.path.normpath(
                os.path.join(
                    self.get_base_path(step_name),
                    dataset_folder,
                    self.get_step_folder_name(step_name),
                )
            )
        except (KeyError, TypeError, ValueError):
            return "<unresolved>"

    def _summary_steps(self) -> List[str]:
        """
        Tabulate each configured step, its class and its output directory.

        :return: One aligned line per step, or an empty list when no step
                 class set is resolved.
        :rtype: list[str]
        """
        steps = (self.data or {}).get("step_class_set", {}).get("steps", {})
        if not steps:
            return []

        name_width = max(len(str(x)) for x in steps)
        class_width = max(len(str(x)) for x in steps.values())
        return [
            f"{name:<{name_width}}  {str(class_name):<{class_width}}  "
            f"{self._step_directory(name)}"
            for name, class_name in steps.items()
        ]

    def _summary_extra(self, width: int) -> List[Tuple[str, List[str]]]:
        """
        Supply subclass-specific summary rows.

        Subclasses override this to report what only they have — the NRT QC
        items, for instance. Rows are inserted after the features row.

        :param width: The total line width the summary is formatted to.
        :type width: int
        :return: ``(label, lines)`` pairs; empty in the base class.
        :rtype: list[tuple[str, list[str]]]
        """
        return []

    def summary(self, width: int = 88) -> str:
        """
        Build a readable summary of what this configuration resolves to.

        This is what :meth:`__str__` returns, so ``print(config)`` shows the
        summary. It reports the source file, the schema status, and — once an
        entry has been selected — the targets, features, input file, row
        filters, and the class and output directory of every step. Nothing is
        recomputed or cached: the summary reflects :attr:`data` as it stands,
        including any changes made to it since :meth:`select`.

        Before an entry is selected, the available entry names are listed
        instead, so a configuration file can be inspected without knowing what
        it contains.

        :param width: The line width to wrap long values to. Step paths are
                      never wrapped, since a broken path is worse than a long
                      line.
        :type width: int
        :return: The summary, as a multi-line string without a trailing
                 newline.
        :rtype: str
        """
        is_valid, message = self.check_schema()
        count = len(self._entry_names())

        rows: List[Tuple[str, List[str]]] = [
            ("source", [self.config_file]),
            (
                "section",
                [f"{self.section_name} ({count} entr{'y' if count == 1 else 'ies'})"],
            ),
            ("schema", self._wrap("valid" if is_valid else message, width)),
        ]

        entry_names = self._entry_names()
        if self.data is None and entry_names:
            rows.append(("entries", self._wrap(", ".join(entry_names), width)))
        elif self.data is not None:
            features = (self.data.get("feature_set") or {}).get("features") or []
            for label, values in (
                ("targets", self._summary_targets()),
                (
                    "features",
                    self._wrap(", ".join(features), width) if features else [],
                ),
                *self._summary_extra(width),
                ("input", self._summary_input_file()),
                ("filters", self._summary_filters(width)),
                ("steps", self._summary_steps()),
            ):
                if values:
                    rows.append((label, values))

        name = self.dataset_name or "<nothing selected>"
        lines = [f"{type(self).__name__}: {name}"]
        for label, values in rows:
            for index, value in enumerate(values):
                indent = (
                    f"  {label:<{self._label_width - 2}}"
                    if index == 0
                    else " " * self._label_width
                )
                lines.append(f"{indent}{value}")

        if self.data is None:
            lines.append(
                "  (call select(<name>) to resolve one of the entries above)"
                if entry_names
                else f"  (no '{self.section_name}' entries found in this file)"
            )

        return "\n".join(lines)

    def __str__(self) -> str:
        """
        Return the configuration summary, so ``print(config)`` is informative.

        :return: The multi-line summary from :meth:`summary`.
        :rtype: str
        """
        return self.summary()

    def __repr__(self) -> str:
        """
        Return a short, single-line representation of the configuration object.

        :return: String identifying the instance, its managed section and the
                 selected entry.
        :rtype: str
        """
        return (
            f"{type(self).__name__}(section_name={self.section_name}, "
            f"dataset_name={self.dataset_name})"
        )
