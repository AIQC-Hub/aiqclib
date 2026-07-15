"""
This module defines the NRTQCConfig class, a specialized configuration handler
for the Near-Real Time Quality Control (NRT QC) module. It extends ConfigBase
to provide structured access and resolution of NRT QC sub-configurations
(QC variable sets, QC item sets, and step class/parameter definitions) from
YAML-based configuration files.
"""

from typing import Dict, List, Optional

from aiqclib.common.base.config_base import ConfigBase
from aiqclib.common.utils.config import get_config_item


class NRTQCConfig(ConfigBase):
    """
    A configuration class for retrieving and organizing settings specific
    to the NRT QC module.

    Extends :class:`aiqclib.common.base.config_base.ConfigBase` by adding logic
    to select NRT QC sets from YAML-based configuration files. The selected set
    references various sub-configurations (QC variable sets, QC item sets, and
    step class definitions). These references are resolved and stored within
    :attr:`data`.

    The resolved ``qc_variable_set`` is also stored under the ``target_set``
    key so the target helpers inherited from :class:`ConfigBase`
    (e.g. :meth:`get_target_names`, :meth:`get_target_file_names`) work
    unchanged for QC variables.
    """

    expected_class_name: str = "NRTQCConfig"
    """
    The class name expected by this configuration to validate it
    aligns with the YAML definition. Used by :class:`aiqclib.common.base.config_base.ConfigBase`.
    """

    def __init__(self, config_file: str, auto_select: bool = False) -> None:
        """
        Initialize a new :class:`NRTQCConfig` instance.

        :param config_file: The path to the YAML file containing NRT QC sets
                            and their sub-configurations.
        :type config_file: str
        :param auto_select: If :obj:`True`, automatically select the first
                            available set from the configuration file.
        :type auto_select: bool
        :raises ValueError: If the YAML is invalid or missing the
                            "nrt_qc_sets" section.
        """
        super().__init__(
            section_name="nrt_qc_sets",
            config_file=config_file,
            auto_select=auto_select,
        )

    def select(self, dataset_name: str) -> None:
        """
        Choose an NRT QC set by name and load its sub-configuration items
        (QC variable sets, QC item sets, step classes and parameters)
        into :attr:`data`.

        :param dataset_name: The name (key) of the desired set in the YAML's
                             "nrt_qc_sets" section.
        :type dataset_name: str
        :raises KeyError: If ``dataset_name`` is not present in the
                          "nrt_qc_sets" section of the YAML, or if a referenced
                          sub-configuration name (e.g. "qc_item_set") is not
                          found in its corresponding top-level section.
        :returns: None
        :rtype: None
        """
        super().select(dataset_name)
        self.data["qc_variable_set"] = get_config_item(
            self.full_config, "qc_variable_sets", self.data["qc_variable_set"]
        )
        self.data["qc_item_set"] = get_config_item(
            self.full_config, "qc_item_sets", self.data["qc_item_set"]
        )
        self.data["step_class_set"] = get_config_item(
            self.full_config, "step_class_sets", self.data["step_class_set"]
        )
        self.data["step_param_set"] = get_config_item(
            self.full_config, "step_param_sets", self.data["step_param_set"]
        )
        # Alias so the ConfigBase target helpers operate on QC variables.
        self.data["target_set"] = self.data["qc_variable_set"]

    def get_qc_items(
        self, default_params: Optional[Dict[str, Dict]] = None
    ) -> List[Dict]:
        """
        Return the enabled QC items with their resolved parameters.

        Each returned item is a dictionary with the keys ``name``, ``params``,
        and ``fail_flag``. Parameters given in the configuration override the
        built-in defaults supplied via ``default_params`` (a shallow, per-key
        merge). ``fail_flag`` defaults to 4 (bad data) when not set.

        :param default_params: Built-in default parameters keyed by item name,
                               typically supplied by the QC item registry.
                               Defaults to :obj:`None` (no defaults).
        :type default_params: Optional[Dict[str, Dict]]
        :return: One dictionary per enabled item, in configuration order.
        :rtype: List[Dict]
        """
        default_params = default_params or {}

        items = []
        for item in self.data["qc_item_set"]["items"]:
            name = item["name"]
            params = {**default_params.get(name, {}), **item.get("params", {})}
            items.append(
                {
                    "name": name,
                    "params": params,
                    "fail_flag": item.get("fail_flag", 4),
                }
            )
        return items

    def get_qc_item_names(self) -> List[str]:
        """
        Return the names of all enabled QC items, in configuration order.

        :return: List of enabled QC item names.
        :rtype: List[str]
        """
        return [x["name"] for x in self.data["qc_item_set"]["items"]]

    def get_variable_flag(self, target_name: str) -> Optional[str]:
        """
        Return the existing NRT QC flag column configured for a variable.

        The flag column is optional and only used by the flag comparison
        step. A flag is treated as absent when the ``flag`` key is missing,
        ``None``, or empty/whitespace (see :meth:`ConfigBase.is_flag_missing`).

        :param target_name: The name of the QC variable.
        :type target_name: str
        :return: The flag column name, or :obj:`None` when no usable flag
                 is configured for the variable.
        :rtype: Optional[str]
        """
        target_value = self.get_target_dict().get(target_name, {})
        if self.is_flag_missing(target_value):
            return None
        return target_value["flag"]
