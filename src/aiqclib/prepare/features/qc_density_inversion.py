"""
This module defines QCDensityInversion, the RTQC14 density inversion test.

The potential density anomaly sigma-0 (UNESCO 1983 algorithm, see
:mod:`aiqclib.common.utils.seawater`) is computed for every observation and
compared at consecutive profile levels in both directions. From top to
bottom, an observation fails when its density is smaller than the previous
(lesser-pressure) density beyond the threshold; from bottom to top, when its
density is larger than the next (greater-pressure) density beyond the
threshold — so both levels of an inverted pair are flagged. Because the
density combines temperature and salinity, both variables are flagged
jointly. Small inversions below the configurable threshold are allowed.
"""

from typing import Dict, List, Tuple

import polars as pl

from aiqclib.common.utils.seawater import sigma0
from aiqclib.prepare.features.qc_item_base import (
    PROFILE_KEYS,
    QCItemFeatureBase,
)

_SIGMA0_COLUMN = "_qc_sigma0"


class QCDensityInversion(QCItemFeatureBase):
    """
    RTQC14 density inversion test (observation-level, temp and psal jointly).

    Fails both levels of a consecutive pair whose sigma-0 difference
    inverts by more than ``threshold`` (0.03 kg/m³ by default). Produces
    ``temp_qc_density_inversion`` and ``psal_qc_density_inversion`` columns
    carrying identical flags. Observations with a missing temperature,
    salinity, or pressure cannot be density-checked and always pass.
    """

    item_name: str = "density_inversion"
    default_params: Dict = {
        "threshold": 0.03,
    }
    scalar_param_names: Tuple[str, ...] = ("threshold",)

    #: Column defining the vertical order of observations within a profile.
    order_column: str = "observation_no"

    def get_variables(self) -> List[str]:
        """
        Return the variables receiving the joint density flag.

        Density combines temperature and salinity, so both are flagged by
        default; an explicit ``col_names`` list restricts the output columns.

        :return: Variable names to produce flag columns for.
        :rtype: List[str]
        """
        info = self.feature_info or {}
        if info.get("col_names"):
            return list(info["col_names"])
        return ["temp", "psal"]

    def compute_flags(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Flag consecutive-level density inversions beyond the threshold.

        :param df: The observations to check (needs temp, psal, and pres).
        :type df: pl.DataFrame
        :return: Observation keys plus the per-variable joint flag columns.
        :rtype: pl.DataFrame
        """
        threshold = self.params["threshold"]

        # sigma-0 is a numpy computation; missing inputs surface as NaN.
        # NaN must become null here: polars orders NaN above every float,
        # while null comparisons stay null and count as a pass.
        with_sigma = df.with_columns(
            pl.Series(
                _SIGMA0_COLUMN, sigma0(df["psal"], df["temp"], df["pres"])
            ).fill_nan(None)
        )

        sigma = pl.col(_SIGMA0_COLUMN)
        prev_sigma = sigma.shift(1).over(PROFILE_KEYS, order_by=self.order_column)
        next_sigma = sigma.shift(-1).over(PROFILE_KEYS, order_by=self.order_column)

        # Top to bottom: this (greater-pressure) level is lighter than the
        # level above. Bottom to top: this (lesser-pressure) level is denser
        # than the level below. Together they flag both levels of a pair.
        fail = (sigma < prev_sigma - threshold) | (sigma > next_sigma + threshold)

        exprs = [
            self._flag_expr(fail, self.flag_column_name(variable))
            for variable in self.get_variables()
        ]
        return self._select_flags(with_sigma, exprs)
