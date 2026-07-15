"""
This module defines QCGradient, the RTQC11 gradient test.

The test fails when the difference between vertically adjacent measurements
is too steep. The test value is ``|V2 - (V3 + V1)/2|`` where V2 is the
tested measurement and V1/V3 its neighbours, compared against
depth-dependent thresholds (temp 9.0/3.0 degC and psal 1.5/0.5 around
500 db by default).
"""

from typing import Dict

import polars as pl

from aiqclib.prepare.features.qc_item_base import QCNeighborStencilBase


class QCGradient(QCNeighborStencilBase):
    """
    RTQC11 gradient test (observation-level, per variable).

    Fails V2 when the gradient test value exceeds the depth-dependent
    threshold. Produces one ``{variable}_qc_gradient`` column per configured
    variable; profile-boundary observations always pass.
    """

    item_name: str = "gradient"
    default_params: Dict = {
        "temp": {"shallow": 9.0, "deep": 3.0},
        "psal": {"shallow": 1.5, "deep": 0.5},
        "depth_threshold": 500,
    }

    def test_value_expr(self, v1: pl.Expr, v2: pl.Expr, v3: pl.Expr) -> pl.Expr:
        """
        The gradient test value: ``|V2 - (V3 + V1)/2|``.

        :param v1: The neighbouring value above V2.
        :type v1: pl.Expr
        :param v2: The value being tested.
        :type v2: pl.Expr
        :param v3: The neighbouring value below V2.
        :type v3: pl.Expr
        :return: The gradient test value expression.
        :rtype: pl.Expr
        """
        return (v2 - (v3 + v1) / 2).abs()
