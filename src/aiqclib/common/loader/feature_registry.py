"""
Module defining the global registry for feature classes.

This module provides ``FEATURE_REGISTRY``, a central mapping of string
identifiers to specific feature-extraction classes within the `aiqclib`
pipeline. Each entry allows for dynamic loading and instantiation of
feature generators based on configuration settings, facilitating the
preparation of datasets by applying various data transformations and
extractions.
"""

from typing import Dict, Type

from aiqclib.common.base.feature_base import FeatureBase
from aiqclib.prepare.features.basic_values import BasicValues
from aiqclib.prepare.features.day_of_year import DayOfYearFeat
from aiqclib.prepare.features.flank_down import FlankDown
from aiqclib.prepare.features.flank_up import FlankUp
from aiqclib.prepare.features.location import LocationFeat
from aiqclib.prepare.features.profile_summary import ProfileSummaryStats
from aiqclib.prepare.features.qc_digit_rollover import QCDigitRollover
from aiqclib.prepare.features.qc_global_range import QCGlobalRange
from aiqclib.prepare.features.qc_gradient import QCGradient
from aiqclib.prepare.features.qc_impossible_date import QCImpossibleDate
from aiqclib.prepare.features.qc_impossible_location import QCImpossibleLocation
from aiqclib.prepare.features.qc_pressure_increasing import QCPressureIncreasing
from aiqclib.prepare.features.qc_regional_range import QCRegionalRange
from aiqclib.prepare.features.qc_spike import QCSpike
from aiqclib.prepare.features.qc_stuck_value import QCStuckValue

#: A dictionary mapping feature identifiers (str) to classes that inherit
#: from :class:`FeatureBase`. These classes are dynamically loaded based
#: on the "feature" key in a feature configuration dictionary.
#:
#: NRT QC items are registered under ``qc_``-prefixed names (the item's
#: short name from the NRT QC configuration prefixed with ``qc_``) so they
#: can also be used as training features in a prepare ``feature_set``.
#:
#: :type: Dict[str, Type[FeatureBase]]
FEATURE_REGISTRY: Dict[str, Type[FeatureBase]] = {
    "location": LocationFeat,
    "day_of_year": DayOfYearFeat,
    "profile_summary_stats": ProfileSummaryStats,
    "basic_values": BasicValues,
    "flank_up": FlankUp,
    "flank_down": FlankDown,
    "qc_impossible_date": QCImpossibleDate,
    "qc_impossible_location": QCImpossibleLocation,
    "qc_global_range": QCGlobalRange,
    "qc_regional_range": QCRegionalRange,
    "qc_pressure_increasing": QCPressureIncreasing,
    "qc_spike": QCSpike,
    "qc_gradient": QCGradient,
    "qc_digit_rollover": QCDigitRollover,
    "qc_stuck_value": QCStuckValue,
}
