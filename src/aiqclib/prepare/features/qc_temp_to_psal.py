"""
This module defines QCTempToPsal, the temperature-to-salinity flag
propagation rule from the NRT QC recommendation document.

When salinity is computed from temperature and conductivity, a temperature
flagged 4 (or 3) corrupts the salinity at the same observation, so the
salinity is flagged 4 (or 3) as well. The item consumes an aggregated
temperature flag column (``temp_nrt_flag`` by default, produced by the
NRT QC flag aggregation step) and emits the propagated salinity flag in
its own column so the propagation stays traceable. Datasets with
independently measured salinity simply omit this item from their
configuration.
"""

from typing import Dict, List, Tuple

import polars as pl

from aiqclib.common.utils.qc_flags import FLAG_BAD, FLAG_GOOD, FLAG_PROBABLY_BAD
from aiqclib.prepare.features.qc_item_base import QCItemFeatureBase


class QCTempToPsal(QCItemFeatureBase):
    """
    Temperature-to-salinity flag propagation (observation-level).

    Copies the aggregated temperature flag onto salinity when it is 3 or 4
    (the propagated flag keeps its severity — ``fail_flag`` does not apply
    to this item). Produces the single column ``psal_qc_temp_to_psal``,
    which is 1 wherever the temperature flag is good or missing.
    """

    item_name: str = "temp_to_psal"
    default_params: Dict = {
        "source_column": "temp_nrt_flag",
        "target_variable": "psal",
    }
    scalar_param_names: Tuple[str, ...] = ("source_column", "target_variable")

    def get_variables(self) -> List[str]:
        """
        Return the single variable receiving the propagated flag.

        :return: The configured target variable (psal by default).
        :rtype: List[str]
        """
        return [self.params["target_variable"]]

    def compute_flags(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Propagate probably-bad/bad temperature flags onto the target
        variable.

        :param df: The observations, including the aggregated temperature
                   flag column (``source_column``).
        :type df: pl.DataFrame
        :return: Observation keys plus the propagated flag column.
        :rtype: pl.DataFrame
        """
        source = pl.col(self.params["source_column"])
        (variable,) = self.get_variables()

        flag = (
            pl.when(source.is_in([FLAG_PROBABLY_BAD, FLAG_BAD]))
            .then(source.cast(pl.Int64))
            .otherwise(pl.lit(FLAG_GOOD, dtype=pl.Int64))
            .alias(self.flag_column_name(variable))
        )
        return self._select_flags(df, [flag])
