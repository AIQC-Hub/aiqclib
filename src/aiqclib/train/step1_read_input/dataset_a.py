"""
This module provides a specialized input class, :class:`InputTrainingSetA`,
designed for reading training and test datasets specific to Copernicus CTD data.
It extends :class:`aiqclib.train.step1_read_input.input_base.InputTrainingSetBase`
to handle particular data configuration and validation requirements.
"""

from aiqclib.common.base.config_base import ConfigBase
from aiqclib.train.step1_read_input.input_base import InputTrainingSetBase


class InputTrainingSetA(InputTrainingSetBase):
    """
    A specialized input class for reading training and test sets for Copernicus CTD data.

    This class extends :class:`aiqclib.train.step1_read_input.input_base.InputTrainingSetBase`
    and provides specific implementations or configurations for handling CTD datasets.
    It sets its :attr:`expected_class_name` to "InputTrainingSetA" so that configuration
    validation in the parent class can correctly match the ``base_class`` value specified in YAML.
    """

    expected_class_name: str = "InputTrainingSetA"

    def __init__(self, config: ConfigBase) -> None:
        """
        Initialize the specialized input training set class with the provided training configuration.

        This constructor calls the parent class's `__init__` method, passing the
        configuration object. It ensures that the base setup for reading training
        and test data is performed.

        :param config: A training configuration object containing paths,
                       file names, and target definitions necessary for
                       data loading and processing.
        :type config: aiqclib.common.base.config_base.ConfigBase
        """
        super().__init__(config=config)
