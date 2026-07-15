"""
This module defines the QCDataSetAll class, the concrete implementation of
the NRT QC module's step 2 (running the configured QC items) for
Copernicus CTD data.
"""

from typing import Optional

import polars as pl

from aiqclib.common.base.config_base import ConfigBase
from aiqclib.nrtqc.step2_run_qc.qc_base import QCDataSetBase


class QCDataSetAll(QCDataSetBase):
    """
    A specialized implementation of
    :class:`aiqclib.nrtqc.step2_run_qc.qc_base.QCDataSetBase` that applies
    the configured QC items to all observations.

    :cvar expected_class_name: The class identifier used for configuration matching.
    :vartype expected_class_name: str
    """

    expected_class_name: str = "QCDataSetAll"

    def __init__(
        self,
        config: ConfigBase,
        input_data: Optional[pl.DataFrame] = None,
    ) -> None:
        """
        Initializes the QCDataSetAll instance with the specified configuration.

        :param config: An NRT QC configuration object.
        :type config: aiqclib.common.base.config_base.ConfigBase
        :param input_data: The validated input observations from step 1.
        :type input_data: Optional[pl.DataFrame]
        """
        super().__init__(config=config, input_data=input_data)
