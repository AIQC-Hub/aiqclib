"""
This module provides centralized registries for various training components,
including dataset readers, model validation strategies, and model-building classes.
Each registry is a dictionary mapping string keys (typically from configuration files)
to their corresponding Python class implementations, facilitating flexible and
extensible model training workflows.
"""

from typing import Dict, Type

from aiqclib.train.step1_read_input.dataset_a import InputTrainingSetA
from aiqclib.train.step1_read_input.input_base import InputTrainingSetBase
from aiqclib.train.step2_validate_model.kfold_validation import KFoldValidation
from aiqclib.train.step2_validate_model.kfold_validation_suite import (
    KFoldValidationSuite,
)
from aiqclib.train.step2_validate_model.validate_base import ValidationBase
from aiqclib.train.step4_build_model.build_model import BuildModel
from aiqclib.train.step4_build_model.build_model_suite import BuildModelSuite
from aiqclib.train.step4_build_model.build_model_base import BuildModelBase

#: Registry mapping string keys to concrete implementations of
#: :class:`~aiqclib.train.step1_read_input.input_base.InputTrainingSetBase`.
#:
#: This dictionary facilitates the dynamic selection of dataset reading classes
#: based on configuration settings, allowing for easy extension and customization
#: of input data handling.
INPUT_TRAINING_SET_REGISTRY: Dict[str, Type[InputTrainingSetBase]] = {
    "InputTrainingSetA": InputTrainingSetA,
}

#: Registry mapping string keys to concrete implementations of
#: :class:`~aiqclib.train.step2_validate_model.validate_base.ValidationBase`.
#:
#: This dictionary allows for the dynamic selection of model validation strategies
#: based on configuration settings, supporting various evaluation methodologies.
MODEL_VALIDATION_REGISTRY: Dict[str, Type[ValidationBase]] = {
    "KFoldValidation": KFoldValidation,
    "KFoldValidationSuite": KFoldValidationSuite,
}

#: Registry mapping string keys to concrete implementations of
#: :class:`~aiqclib.train.step4_build_model.build_model_base.BuildModelBase`.
#:
#: This dictionary enables the dynamic selection of model-building classes
#: based on configuration settings, providing flexibility in choosing and
#: configuring different model architectures.
BUILD_MODEL_REGISTRY: Dict[str, Type[BuildModelBase]] = {
    "BuildModel": BuildModel,
    "BuildModelSuite": BuildModelSuite,
}
