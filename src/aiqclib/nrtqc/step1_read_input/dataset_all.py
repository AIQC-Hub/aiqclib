"""
This module defines the InputDataSetAll class for the NRT QC module, which
is responsible for loading and preparing Copernicus CTD input data. It
extends InputDataSetBase, reusing the shared reading, column validation,
and preprocessing machinery of the preparation pipeline.
"""

from aiqclib.common.base.config_base import ConfigBase
from aiqclib.prepare.step1_read_input.input_base import InputDataSetBase


class InputDataSetAll(InputDataSetBase):
    """
    A specialized implementation of :class:`aiqclib.prepare.step1_read_input.input_base.InputDataSetBase`
    for reading the NRT QC input data.

    This class serves as a concrete implementation for reading and validating
    input data as defined by the NRT QC configuration schema. Row filtering
    is disabled by default in the NRT QC template (near-real time data is
    checked as-is).

    :cvar expected_class_name: The class identifier used for configuration matching.
    :vartype expected_class_name: str
    """

    expected_class_name: str = "InputDataSetAll"

    def __init__(self, config: ConfigBase) -> None:
        """
        Initializes the InputDataSetAll instance with the specified configuration.

        :param config: A configuration object containing paths and parameters
                       necessary for retrieving the NRT QC input data.
        :type config: aiqclib.common.base.config_base.ConfigBase
        """
        super().__init__(config=config)
