"""
aiqclib Interface Module
========================

This module provides a high-level interface to the aiqclib library,
exposing core functionalities for configuration management, dataset
preparation, model training and evaluation, and dataset classification.

The public functions are imported lazily (PEP 562): ``import aiqclib`` or
importing a submodule such as ``aiqclib.prepare.features.qc_spike`` does not
load the training and classification stack (xgboost, scikit-learn,
matplotlib) until one of the functions below is first used.

Attributes:
    __version__ (str): The version of the aiqclib library.
"""

from importlib import import_module
from importlib.metadata import version, PackageNotFoundError
from typing import TYPE_CHECKING, Any, Dict, List

try:
    __version__ = version("aiqclib")
except PackageNotFoundError:
    __version__ = "unknown"

#: Maps each public name to the module that defines it.
_LAZY_ATTRIBUTES: Dict[str, str] = {
    "available_modes": "aiqclib.interface.batch",
    "run_batch": "aiqclib.interface.batch",
    "classify_dataset": "aiqclib.interface.classify",
    "read_config": "aiqclib.interface.config",
    "read_config_template": "aiqclib.interface.config",
    "write_config_template": "aiqclib.interface.config",
    "create_training_dataset": "aiqclib.interface.prepare",
    "read_shap_scores": "aiqclib.interface.shap_io",
    "format_summary_stats": "aiqclib.interface.stats",
    "get_summary_stats": "aiqclib.interface.stats",
    "train_and_evaluate": "aiqclib.interface.train",
}

__all__ = [
    "available_modes",
    "classify_dataset",
    "read_config",
    "read_config_template",
    "write_config_template",
    "create_training_dataset",
    "format_summary_stats",
    "get_summary_stats",
    "read_shap_scores",
    "run_batch",
    "train_and_evaluate",
]

if TYPE_CHECKING:  # pragma: no cover - for type checkers and IDEs only
    from aiqclib.interface.batch import available_modes, run_batch
    from aiqclib.interface.classify import classify_dataset
    from aiqclib.interface.config import read_config
    from aiqclib.interface.config import read_config_template
    from aiqclib.interface.config import write_config_template
    from aiqclib.interface.prepare import create_training_dataset
    from aiqclib.interface.shap_io import read_shap_scores
    from aiqclib.interface.stats import format_summary_stats
    from aiqclib.interface.stats import get_summary_stats
    from aiqclib.interface.train import train_and_evaluate


def __getattr__(name: str) -> Any:
    """
    Import a public function on first access and cache it on the package.

    :param name: The attribute being looked up.
    :type name: str
    :raises AttributeError: If ``name`` is not a public name of the package.
    :return: The requested function.
    :rtype: Any
    """
    module_name = _LAZY_ATTRIBUTES.get(name)
    if module_name is None:
        raise AttributeError(f"module 'aiqclib' has no attribute '{name}'")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> List[str]:
    """
    List the package attributes, including the not yet imported public names.

    :return: The sorted attribute names.
    :rtype: list[str]
    """
    return sorted(set(globals()) | set(__all__))
