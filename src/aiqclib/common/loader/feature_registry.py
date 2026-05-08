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

#: A dictionary mapping feature identifiers (str) to classes that inherit
#: from :class:`FeatureBase`. These classes are dynamically loaded based
#: on the "feature" key in a feature configuration dictionary.
#:
#: :type: Dict[str, Type[FeatureBase]]
FEATURE_REGISTRY: Dict[str, Type[FeatureBase]] = {
    "location": LocationFeat,
    "day_of_year": DayOfYearFeat,
    "profile_summary_stats": ProfileSummaryStats,
    "basic_values": BasicValues,
    "flank_up": FlankUp,
    "flank_down": FlankDown,
}
