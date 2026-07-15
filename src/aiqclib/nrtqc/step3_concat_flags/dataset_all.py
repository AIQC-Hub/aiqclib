"""
This module defines the ConcatDataSetAll class, the concrete implementation
of the NRT QC module's step 3 (flag aggregation and output) for Copernicus
CTD data.
"""

from typing import Optional

import polars as pl

from aiqclib.common.base.config_base import ConfigBase
from aiqclib.nrtqc.step3_concat_flags.concat_base import ConcatFlagsBase


class ConcatDataSetAll(ConcatFlagsBase):
    """
    A specialized implementation of
    :class:`aiqclib.nrtqc.step3_concat_flags.concat_base.ConcatFlagsBase`
    that aggregates the QC item flags of all observations.

    :cvar expected_class_name: The class identifier used for configuration matching.
    :vartype expected_class_name: str
    """

    expected_class_name: str = "ConcatDataSetAll"

    def __init__(
        self,
        config: ConfigBase,
        qc_data: Optional[pl.DataFrame] = None,
    ) -> None:
        """
        Initializes the ConcatDataSetAll instance with the specified configuration.

        :param config: An NRT QC configuration object.
        :type config: aiqclib.common.base.config_base.ConfigBase
        :param qc_data: The flag frame produced by step 2.
        :type qc_data: Optional[pl.DataFrame]
        """
        super().__init__(config=config, qc_data=qc_data)
