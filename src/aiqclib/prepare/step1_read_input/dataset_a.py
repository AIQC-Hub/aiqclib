"""
This module provides the InputDataSetA class, a specific implementation for
reading and preparing Copernicus CTD data.

The module extends the base functionality provided by InputDataSetBase to
implement concrete logic for data retrieval and initial processing as part
of the data preparation pipeline.
"""

from aiqclib.common.base.config_base import ConfigBase
from aiqclib.prepare.step1_read_input.input_base import InputDataSetBase


class InputDataSetA(InputDataSetBase):
    """
    A subclass of :class:`InputDataSetBase` providing specific logic to read
    Copernicus CTD data.

    This class ensures compatibility with YAML configuration files by defining
    the expected class name used during the dynamic instantiation process.

    :ivar expected_class_name: String identifier used to match configuration keys.
    :vartype expected_class_name: str
    """

    expected_class_name: str = "InputDataSetA"

    def __init__(self, config: ConfigBase) -> None:
        """
        Initialize the Copernicus CTD input dataset.

        :param config: The dataset configuration object, which includes paths
                       and parameters for retrieving Copernicus CTD data.
        :type config: ConfigBase
        :return: None
        :rtype: NoneType
        """
        super().__init__(config=config)
