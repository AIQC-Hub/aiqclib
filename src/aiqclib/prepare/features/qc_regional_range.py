"""
This module defines QCRegionalRange, the RTQC7 regional range test.

The same range check as the global range test, but with the ranges of the
configuration file's region (one configuration file is prepared per region,
so no polygon membership test is needed). There are no built-in default
ranges: the region's bounds must be supplied in the configuration, and a
regional_range item without them raises an error rather than silently
passing everything.
"""

from typing import Dict

from aiqclib.prepare.features.qc_global_range import QCGlobalRange


class QCRegionalRange(QCGlobalRange):
    """
    RTQC7 regional range test (observation-level, per variable).

    Identical to :class:`QCGlobalRange` except that the ranges come from
    the region of the active configuration file and have no defaults.
    Produces one ``{variable}_qc_regional_range`` column per configured
    variable.
    """

    item_name: str = "regional_range"
    default_params: Dict = {}
