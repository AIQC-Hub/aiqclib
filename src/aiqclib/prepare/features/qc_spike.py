"""
This module defines QCSpike, the RTQC9 spike test.

A spike is a measurement quite different from its vertical neighbours in
both size and gradient. The test value is
``|V2 - (V3 + V1)/2| - |(V3 - V1)/2|`` where V2 is the tested measurement
and V1/V3 its neighbours, compared against depth-dependent thresholds
(temp 6.0/2.0 degC and psal 0.9/0.3 around 500 db by default).
"""

from typing import Dict

import polars as pl

from aiqclib.prepare.features.qc_item_base import QCNeighborStencilBase


class QCSpike(QCNeighborStencilBase):
    """
    RTQC9 spike test (observation-level, per variable).

    Fails V2 when the spike test value exceeds the depth-dependent
    threshold. Produces one ``{variable}_qc_spike`` column per configured
    variable; profile-boundary observations always pass.
    """

    item_name: str = "spike"
    default_params: Dict = {
        "temp": {"shallow": 6.0, "deep": 2.0},
        "psal": {"shallow": 0.9, "deep": 0.3},
        "depth_threshold": 500,
    }

    def test_value_expr(self, v1: pl.Expr, v2: pl.Expr, v3: pl.Expr) -> pl.Expr:
        """
        The spike test value: ``|V2 - (V3 + V1)/2| - |(V3 - V1)/2|``.

        :param v1: The neighbouring value above V2.
        :type v1: pl.Expr
        :param v2: The value being tested.
        :type v2: pl.Expr
        :param v3: The neighbouring value below V2.
        :type v3: pl.Expr
        :return: The spike test value expression.
        :rtype: pl.Expr
        """
        return (v2 - (v3 + v1) / 2).abs() - ((v3 - v1) / 2).abs()
