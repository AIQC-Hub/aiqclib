"""
This module provides the QCDataSetBase class, the core of the NRT QC
module's step 2: applying the configured QC items to the input data.

Each enabled QC item is resolved through the feature registry (the items
are ordinary feature classes under ``prepare/features``, registered with
``qc_``-prefixed names), instantiated with its configured parameters, and
run over the full input frame. The produced flag columns are joined back
onto the data, yielding one column per item/variable combination.
"""

import os
from typing import Dict, List, Optional

import polars as pl

from aiqclib.common.base.config_base import ConfigBase
from aiqclib.common.base.dataset_base import DataSetBase
from aiqclib.common.loader.feature_registry import FEATURE_REGISTRY
from aiqclib.prepare.features.qc_item_base import OBSERVATION_KEYS

#: QC items that need the aggregated per-variable flags and therefore run
#: during the flag aggregation step (step 3) instead of step 2.
DEFERRED_QC_ITEMS: tuple = ("temp_to_psal",)


class QCDataSetBase(DataSetBase):
    """
    Base class for running the configured NRT QC items (step ``"qc"``).

    Takes the validated input data from step 1 and applies every item
    enabled in the configuration's ``qc_item_set`` — except the deferred
    propagation items, which need the aggregated flags of step 3. The
    result, stored in :attr:`qc_data`, is the input frame plus one flag
    column per item/variable combination.
    """

    def __init__(
        self,
        config: ConfigBase,
        input_data: Optional[pl.DataFrame] = None,
    ) -> None:
        """
        Initialize the QC step with the configuration and input data.

        :param config: An NRT QC configuration object.
        :type config: aiqclib.common.base.config_base.ConfigBase
        :param input_data: The validated input observations from step 1.
        :type input_data: Optional[pl.DataFrame]
        """
        super().__init__(step_name="qc", config=config)

        #: The default name for the intermediate flag file.
        self.default_file_name: str = "nrt_qc_flags.parquet"

        #: The resolved output path for the intermediate flag file.
        self.output_file_name: str = self.config.get_full_file_name(
            step_name="qc", default_file_name=self.default_file_name
        )

        self.input_data: Optional[pl.DataFrame] = input_data
        #: The input data with all QC item flag columns appended.
        self.qc_data: Optional[pl.DataFrame] = None

    def run_qc_items(self) -> None:
        """
        Apply every enabled QC item and append its flag columns.

        Items are applied in configuration order. Each item receives the
        raw input frame (items are independent of each other's flags), and
        its flag columns are joined onto the accumulating result by the
        observation keys. Deferred items (:data:`DEFERRED_QC_ITEMS`) are
        skipped here and handled by step 3.

        :raises ValueError: If an enabled item has no registered feature
                            class (``qc_{name}`` missing from the registry).
        """
        result = self.input_data

        for item in self.config.get_qc_items():
            if item["name"] in DEFERRED_QC_ITEMS:
                continue
            flags = self._run_item(item)
            result = result.join(flags, on=OBSERVATION_KEYS, maintain_order="left")

        self.qc_data = result

    def _run_item(self, item: Dict) -> pl.DataFrame:
        """
        Run a single QC item over the input data.

        :param item: A resolved item entry from
                     :meth:`NRTQCConfig.get_qc_items` (name, params,
                     fail_flag).
        :type item: Dict
        :return: The observation keys plus the item's flag column(s).
        :rtype: pl.DataFrame
        :raises ValueError: If the item has no registered feature class.
        """
        registry_name = f"qc_{item['name']}"
        feature_class = FEATURE_REGISTRY.get(registry_name)
        if feature_class is None:
            raise ValueError(
                f"Unknown QC item '{item['name']}': no feature class "
                f"'{registry_name}' registered."
            )

        ds = feature_class(
            feature_info={
                "params": item["params"],
                "fail_flag": item["fail_flag"],
            },
            filtered_input=self.input_data,
        )
        ds.extract_features()
        return ds.features

    def qc_item_columns(self) -> List[str]:
        """
        Return the flag columns added by :meth:`run_qc_items`.

        :return: The columns of :attr:`qc_data` absent from the input.
        :rtype: List[str]
        :raises ValueError: If :meth:`run_qc_items` has not been run yet.
        """
        if self.qc_data is None:
            raise ValueError("Member variable 'qc_data' must not be empty.")
        return [c for c in self.qc_data.columns if c not in self.input_data.columns]

    def write_qc_data(self) -> None:
        """
        Write the intermediate flag frame to a Parquet file.

        :raises ValueError: If :attr:`qc_data` is empty, indicating the QC
                            items have not been run.
        """
        if self.qc_data is None:
            raise ValueError("Member variable 'qc_data' must not be empty.")

        os.makedirs(os.path.dirname(self.output_file_name), exist_ok=True)
        self.qc_data.write_parquet(self.output_file_name)
