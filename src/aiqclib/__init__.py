"""
aiqclib Interface Module
========================

This module provides a high-level interface to the aiqclib library,
exposing core functionalities for configuration management, dataset
preparation, model training and evaluation, and dataset classification.

Attributes:
    __version__ (str): The version of the aiqclib library.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("aiqclib")
except PackageNotFoundError:
    __version__ = "unknown"

from aiqclib.interface.classify import classify_dataset
from aiqclib.interface.config import read_config
from aiqclib.interface.config import write_config_template
from aiqclib.interface.prepare import create_training_dataset
from aiqclib.interface.stats import format_summary_stats
from aiqclib.interface.stats import get_summary_stats
from aiqclib.interface.train import train_and_evaluate

__all__ = [
    "classify_dataset",
    "read_config",
    "write_config_template",
    "create_training_dataset",
    "format_summary_stats",
    "get_summary_stats",
    "train_and_evaluate",
]
