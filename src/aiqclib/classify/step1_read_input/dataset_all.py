"""
This module defines the InputDataSetAll class, which is responsible for
loading and preparing a specific combination of input datasets, namely
Copernicus CTD data. It extends InputDataSetBase and leverages
a configuration object to manage data retrieval and processing.
"""

from aiqclib.common.base.config_base import ConfigBase
from aiqclib.prepare.step1_read_input.input_base import InputDataSetBase


class InputDataSetAll(InputDataSetBase):
    """
    A specialized implementation of :class:`aiqclib.prepare.step1_read_input.input_base.InputDataSetBase`
    designed for handling Copernicus CTD data.

    This class serves as a concrete implementation for reading and validating
    input data as defined by the application's configuration schema.

    :cvar expected_class_name: The class identifier used for configuration matching.
    :vartype expected_class_name: str
    """

    expected_class_name: str = "InputDataSetAll"

    def __init__(self, config: ConfigBase) -> None:
        """
        Initializes the InputDataSetAll instance with the specified configuration.

        This constructor propagates the configuration object to the base class
        to ensure paths and parameters are correctly initialized for data retrieval.

        :param config: A configuration object containing paths and parameters necessary
                       for retrieving Copernicus CTD data.
        :type config: aiqclib.common.base.config_base.ConfigBase
        """
        super().__init__(config=config)
