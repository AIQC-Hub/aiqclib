"""
This module provides factory functions to dynamically load and instantiate
the dataset processing steps of the NRT QC pipeline.

It uses an NRT QC configuration object
(:class:`aiqclib.common.config.nrtqc_config.NRTQCConfig`) to determine which
implementation should be loaded for each step, relying on the registries in
:mod:`aiqclib.common.loader.nrtqc_registry` to map class names from the
configuration to their Python types.
"""

from typing import Dict, Optional, Type

import polars as pl

from aiqclib.common.base.config_base import ConfigBase
from aiqclib.common.base.dataset_base import DataSetBase
from aiqclib.common.loader.nrtqc_registry import (
    COMPARE_NRTQC_REGISTRY,
    CONCAT_NRTQC_REGISTRY,
    INPUT_NRTQC_REGISTRY,
    QC_NRTQC_REGISTRY,
)
from aiqclib.nrtqc.step2_run_qc.qc_base import QCDataSetBase
from aiqclib.nrtqc.step3_concat_flags.concat_base import ConcatFlagsBase
from aiqclib.nrtqc.step4_compare_flags.compare_base import CompareFlagsBase
from aiqclib.prepare.step1_read_input.input_base import InputDataSetBase


def _get_nrtqc_class(
    config: ConfigBase, step: str, registry: Dict[str, Type[DataSetBase]]
) -> Type[DataSetBase]:
    """Retrieve the class constructor from the specified registry for a step.

    :param config: An NRT QC configuration object that contains the base
                   class name for the requested step in the YAML.
    :type config: ConfigBase
    :param step: The step name defined in the YAML (e.g. "input" or "qc").
    :type step: str
    :param registry: A dictionary mapping class names to dataset class types.
    :type registry: Dict[str, Type[DataSetBase]]
    :raises ValueError: If the class name from the configuration cannot be
                        found in the given registry.
    :returns: The class constructor associated with the requested step.
    :rtype: Type[DataSetBase]
    """
    class_name = config.get_base_class(step)
    dataset_class = registry.get(class_name)
    if not dataset_class:
        raise ValueError(f"Unknown NRT QC class specified: {class_name}")

    return dataset_class


def load_nrtqc_step1_input_dataset(config: ConfigBase) -> InputDataSetBase:
    """Instantiate the configured step 1 (read input) class.

    :param config: The NRT QC configuration object.
    :type config: ConfigBase
    :returns: An instance of the configured input dataset class.
    :rtype: InputDataSetBase
    """
    dataset_class = _get_nrtqc_class(config, "input", INPUT_NRTQC_REGISTRY)
    return dataset_class(config=config)


def load_nrtqc_step2_qc_dataset(
    config: ConfigBase, input_data: Optional[pl.DataFrame] = None
) -> QCDataSetBase:
    """Instantiate the configured step 2 (run QC items) class.

    :param config: The NRT QC configuration object.
    :type config: ConfigBase
    :param input_data: The validated input observations from step 1.
    :type input_data: Optional[pl.DataFrame]
    :returns: An instance of the configured QC dataset class.
    :rtype: QCDataSetBase
    """
    dataset_class = _get_nrtqc_class(config, "qc", QC_NRTQC_REGISTRY)
    return dataset_class(config=config, input_data=input_data)


def load_nrtqc_step3_concat_dataset(
    config: ConfigBase, qc_data: Optional[pl.DataFrame] = None
) -> ConcatFlagsBase:
    """Instantiate the configured step 3 (aggregate flags) class.

    :param config: The NRT QC configuration object.
    :type config: ConfigBase
    :param qc_data: The flag frame produced by step 2.
    :type qc_data: Optional[pl.DataFrame]
    :returns: An instance of the configured flag aggregation class.
    :rtype: ConcatFlagsBase
    """
    dataset_class = _get_nrtqc_class(config, "concat", CONCAT_NRTQC_REGISTRY)
    return dataset_class(config=config, qc_data=qc_data)


def load_nrtqc_step4_compare_dataset(
    config: ConfigBase, merged_data: Optional[pl.DataFrame] = None
) -> CompareFlagsBase:
    """Instantiate the configured step 4 (compare flags) class.

    :param config: The NRT QC configuration object.
    :type config: ConfigBase
    :param merged_data: The final frame produced by step 3.
    :type merged_data: Optional[pl.DataFrame]
    :returns: An instance of the configured flag comparison class.
    :rtype: CompareFlagsBase
    """
    dataset_class = _get_nrtqc_class(config, "compare", COMPARE_NRTQC_REGISTRY)
    return dataset_class(config=config, merged_data=merged_data)
