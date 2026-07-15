"""
This module defines the CompareFlagsAll class, the concrete implementation
of the NRT QC module's step 4 (existing-vs-new flag comparison) for
Copernicus CTD data.
"""

from typing import Optional

import polars as pl

from aiqclib.common.base.config_base import ConfigBase
from aiqclib.nrtqc.step4_compare_flags.compare_base import CompareFlagsBase


class CompareFlagsAll(CompareFlagsBase):
    """
    A specialized implementation of
    :class:`aiqclib.nrtqc.step4_compare_flags.compare_base.CompareFlagsBase`
    that compares the flags of all observations.

    :cvar expected_class_name: The class identifier used for configuration matching.
    :vartype expected_class_name: str
    """

    expected_class_name: str = "CompareFlagsAll"

    def __init__(
        self,
        config: ConfigBase,
        merged_data: Optional[pl.DataFrame] = None,
    ) -> None:
        """
        Initializes the CompareFlagsAll instance with the specified configuration.

        :param config: An NRT QC configuration object.
        :type config: aiqclib.common.base.config_base.ConfigBase
        :param merged_data: The final frame produced by step 3.
        :type merged_data: Optional[pl.DataFrame]
        """
        super().__init__(config=config, merged_data=merged_data)
