"""
This module provides the ConcatFlagsBase class, the core of the NRT QC
module's step 3: aggregating the per-item flag columns into a final NRT
flag per variable and writing the module's output.

For each configured variable, the final ``{variable}_nrt_flag`` is the
most severe flag among that variable's item columns plus the profile-level
(variable-independent) item columns. The deferred temperature-to-salinity
propagation item runs afterwards, on the aggregated temperature flag, and
raises the salinity flag where needed. The output parquet holds all
original input columns, every item column, and the final NRT flags.
"""

import os
from typing import Dict, List, Optional

import polars as pl

from aiqclib.common.base.config_base import ConfigBase
from aiqclib.common.base.dataset_base import DataSetBase
from aiqclib.common.loader.feature_registry import FEATURE_REGISTRY
from aiqclib.common.utils.qc_flags import FLAG_GOOD, worst_flag
from aiqclib.prepare.features.qc_item_base import OBSERVATION_KEYS
from aiqclib.nrtqc.step2_run_qc.qc_base import DEFERRED_QC_ITEMS


class ConcatFlagsBase(DataSetBase):
    """
    Base class for aggregating QC item flags (step ``"concat"``).

    Takes the flag frame produced by step 2 and computes the final
    ``{variable}_nrt_flag`` columns, applying the deferred propagation
    items (e.g. ``temp_to_psal``) last. The result is stored in
    :attr:`merged_data` and written by :meth:`write_merged_data`.
    """

    def __init__(
        self,
        config: ConfigBase,
        qc_data: Optional[pl.DataFrame] = None,
    ) -> None:
        """
        Initialize the aggregation step with the configuration and flags.

        :param config: An NRT QC configuration object.
        :type config: aiqclib.common.base.config_base.ConfigBase
        :param qc_data: The flag frame produced by step 2 (input columns
                        plus per-item flag columns).
        :type qc_data: Optional[pl.DataFrame]
        """
        super().__init__(step_name="concat", config=config)

        #: The default name for the final output file.
        self.default_file_name: str = "nrt_qc_output.parquet"

        #: The resolved output path for the final output file.
        self.output_file_name: str = self.config.get_full_file_name(
            step_name="concat", default_file_name=self.default_file_name
        )

        self.qc_data: Optional[pl.DataFrame] = qc_data
        #: The final frame: input columns, item columns, and NRT flags.
        self.merged_data: Optional[pl.DataFrame] = None

    def _flag_columns_for(self, df: pl.DataFrame, variable: str) -> List[str]:
        """
        Return the item flag columns applicable to a variable.

        These are the variable's own item columns
        (``{variable}_qc_{item}``) plus the profile-level columns
        (``qc_{item}``), restricted to enabled items present in the frame.

        :param df: The flag frame.
        :type df: pl.DataFrame
        :param variable: The variable to aggregate.
        :type variable: str
        :return: The applicable flag column names.
        :rtype: List[str]
        """
        columns = []
        for item_name in self.config.get_qc_item_names():
            for candidate in (f"{variable}_qc_{item_name}", f"qc_{item_name}"):
                if candidate in df.columns:
                    columns.append(candidate)
        return columns

    def concat_flags(self) -> None:
        """
        Compute the final NRT flag per variable and apply propagation.

        For every configured variable, ``{variable}_nrt_flag`` is the most
        severe flag among its applicable item columns (or good when no
        item produced a column for it). Deferred propagation items run
        last, on the aggregated flags.

        :raises ValueError: If :attr:`qc_data` is empty.
        """
        if self.qc_data is None:
            raise ValueError("Member variable 'qc_data' must not be empty.")

        df = self.qc_data
        for variable in self.config.get_target_names():
            flag_columns = self._flag_columns_for(df, variable)
            final = (
                worst_flag(*flag_columns)
                if flag_columns
                else pl.lit(FLAG_GOOD, dtype=pl.Int64)
            )
            df = df.with_columns(final.alias(f"{variable}_nrt_flag"))

        for item in self.config.get_qc_items():
            if item["name"] in DEFERRED_QC_ITEMS:
                df = self._apply_propagation(df, item)

        self.merged_data = df

    def _apply_propagation(self, df: pl.DataFrame, item: Dict) -> pl.DataFrame:
        """
        Run a deferred propagation item and raise the target's final flag.

        The item consumes the aggregated source flag (e.g.
        ``temp_nrt_flag``), emits its own traceable column (e.g.
        ``psal_qc_temp_to_psal``), and the target's ``{variable}_nrt_flag``
        is raised to the propagated severity where needed.

        :param df: The frame with the aggregated NRT flags.
        :type df: pl.DataFrame
        :param item: A resolved item entry from
                     :meth:`NRTQCConfig.get_qc_items`.
        :type item: Dict
        :return: The frame with the propagation column and updated flag.
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
            filtered_input=df,
        )
        ds.extract_features()
        df = df.join(ds.features, on=OBSERVATION_KEYS, maintain_order="left")

        (target_variable,) = ds.get_variables()
        propagated_column = ds.flag_column_name(target_variable)
        final_column = f"{target_variable}_nrt_flag"

        flag_columns = [propagated_column]
        if final_column in df.columns:
            flag_columns.append(final_column)
        return df.with_columns(worst_flag(*flag_columns).alias(final_column))

    def write_merged_data(self) -> None:
        """
        Write the final output frame to a Parquet file.

        :raises ValueError: If :attr:`merged_data` is empty, indicating the
                            flags have not been aggregated.
        """
        if self.merged_data is None:
            raise ValueError("Member variable 'merged_data' must not be empty.")

        os.makedirs(os.path.dirname(self.output_file_name), exist_ok=True)
        self.merged_data.write_parquet(self.output_file_name)
