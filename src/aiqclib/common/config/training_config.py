"""
This module defines the `TrainingConfig` class, which is responsible for
managing and accessing training-related configurations from a YAML file.

It extends :class:`aiqclib.common.base.config_base.ConfigBase` to provide
structured access to dataset settings, including targets, step classes,
and step parameters, by resolving references within the configuration.
"""

from aiqclib.common.base.config_base import ConfigBase
from aiqclib.common.utils.config import get_config_item


class TrainingConfig(ConfigBase):
    """
    A configuration class providing interfaces for training dataset settings.

    Inherits from :class:`ConfigBase` with an expectation of working
    under the "training_sets" section in the YAML configuration.
    Leverages methods like :meth:`select` to initialize and fetch
    subset configurations (e.g., target sets, step parameters).

    .. note::

       :attr:`expected_class_name` must match the YAML's ``base_class``
       property if you intend to instantiate this class directly from config.
    """

    expected_class_name: str = "TrainingConfig"
    """
    The class name expected by :class:`ConfigBase` for consistency checks
    when instantiating TrainingConfig from YAML.
    """

    def __init__(self, config_file: str, auto_select: bool = False) -> None:
        """
        Initialize the training configuration.

        :param config_file: The path to the YAML configuration file.
                            This file will be used to load the training sets.
        :type config_file: str
        :param auto_select: If True, automatically selects the first dataset
                            found in the `training_sets` section upon initialization.
        :type auto_select: bool
        :raises ValueError: If the YAML structure is invalid
                            or `training_sets` cannot be found.
        """
        super().__init__(
            section_name="training_sets",
            config_file=config_file,
            auto_select=auto_select,
        )

    def select(self, dataset_name: str) -> None:
        """
        Select a named dataset from the `training_sets` configuration,
        retrieving nested configurations for targets, step classes,
        and step parameters.

        After calling :meth:`select`, sub-keys (``target_set``,
        ``step_class_set``, etc.) are populated from their respective
        config dictionaries by resolving their references within the full configuration.

        :param dataset_name: The key name of the dataset to select
                             within :attr:`data` (which references
                             the `training_sets` section).
        :type dataset_name: str
        :raises KeyError: If ``dataset_name`` is not found within
                          the `training_sets` dictionary.
        """
        super().select(dataset_name)
        self.data["target_set"] = get_config_item(
            self.full_config, "target_sets", self.data["target_set"]
        )
        self.data["step_class_set"] = get_config_item(
            self.full_config, "step_class_sets", self.data["step_class_set"]
        )
        self.data["step_param_set"] = get_config_item(
            self.full_config, "step_param_sets", self.data["step_param_set"]
        )
