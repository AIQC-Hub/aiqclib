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

from aiqclib.interface.batch import available_modes, run_batch
from aiqclib.interface.classify import classify_dataset
from aiqclib.interface.config import read_config
from aiqclib.interface.config import write_config_template
from aiqclib.interface.nrtqc import run_nrt_qc
from aiqclib.interface.prepare import create_training_dataset
from aiqclib.interface.shap_io import read_shap_scores
from aiqclib.interface.stats import format_summary_stats
from aiqclib.interface.stats import get_summary_stats
from aiqclib.interface.train import train_and_evaluate

__all__ = [
    "available_modes",
    "classify_dataset",
    "read_config",
    "write_config_template",
    "create_training_dataset",
    "format_summary_stats",
    "get_summary_stats",
    "read_shap_scores",
    "run_batch",
    "run_nrt_qc",
    "train_and_evaluate",
]
