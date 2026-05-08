"""
This module defines the ClassificationConfig class, a specialized configuration handler
for managing dataset-related settings pertinent to machine learning classification tasks.
It extends ConfigBase to provide structured access and resolution of various
sub-configurations (e.g., target sets, feature sets, step class definitions)
from YAML-based configuration files, simplifying the management of complex
ML pipeline configurations.
"""

from aiqclib.common.base.config_base import ConfigBase
from aiqclib.common.utils.config import get_config_item


class ClassificationConfig(ConfigBase):
    """
    A configuration class for retrieving and organizing dataset-related
    configurations specific to classification tasks.

    Extends :class:`aiqclib.common.base.config_base.ConfigBase` by adding logic to select datasets
    from YAML-based configuration files. The selected dataset references
    various sub-configurations (e.g., target sets, feature sets, and
    step class definitions). These references are resolved and stored
    within :attr:`data`.
    """

    expected_class_name: str = "ClassificationConfig"
    """
    The class name expected by this configuration to validate it 
    aligns with the YAML definition. Used by :class:`aiqclib.common.base.config_base.ConfigBase`.
    """

    def __init__(self, config_file: str, auto_select: bool = False) -> None:
        """
        Initialize a new :class:`ClassificationConfig` instance.

        :param config_file: The path to the YAML file containing
                            classification datasets and their sub-configurations.
        :type config_file: str
        :param auto_select: If :obj:`True`, automatically select the first
                            available dataset from the configuration file.
        :type auto_select: bool
        :raises ValueError: If the YAML is invalid or missing the
                            "classification_sets" section.
        """
        super().__init__(
            section_name="classification_sets",
            config_file=config_file,
            auto_select=auto_select,
        )

    def select(self, dataset_name: str) -> None:
        """
        Choose a dataset by name and load its sub-configuration items
        (e.g., target sets, feature sets) into :attr:`data`.

        This method retrieves multiple related configurations by calling
        :func:`aiqclib.common.utils.config.get_config_item` on relevant
        sections of the YAML file. It expects that the initial `self.data`
        population from `super().select` contains references to these
        sub-configurations, which are then resolved.

        :param dataset_name: The name (key) of the desired dataset
                             in the YAML's "classification_sets" dictionary.
        :type dataset_name: str
        :raises KeyError: If ``dataset_name`` is not present in the
                          "classification_sets" section of the YAML,
                          or if a referenced sub-configuration name (e.g.,
                          "target_set" within the selected dataset) is not
                          found in its corresponding top-level section
                          (e.g., "target_sets"), or if any of the required
                          sub-configuration keys (e.g., "target_set",
                          "feature_set") are missing from the selected
                          dataset configuration itself.
        :returns: None
        :rtype: None
        """
        super().select(dataset_name)
        self.data["target_set"] = get_config_item(
            self.full_config, "target_sets", self.data["target_set"]
        )
        self.data["summary_stats_set"] = get_config_item(
            self.full_config, "summary_stats_sets", self.data["summary_stats_set"]
        )
        self.data["feature_set"] = get_config_item(
            self.full_config, "feature_sets", self.data["feature_set"]
        )
        self.data["feature_param_set"] = get_config_item(
            self.full_config, "feature_param_sets", self.data["feature_param_set"]
        )
        self.data["feature_stats_set"] = get_config_item(
            self.full_config, "feature_stats_sets", self.data["feature_stats_set"]
        )
        self.data["step_class_set"] = get_config_item(
            self.full_config, "step_class_sets", self.data["step_class_set"]
        )
        self.data["step_param_set"] = get_config_item(
            self.full_config, "step_param_sets", self.data["step_param_set"]
        )
        self.update_feature_param_with_stats()
