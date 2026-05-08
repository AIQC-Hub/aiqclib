"""
This module provides a comprehensive registry of model classes that can be used
during training or inference steps.

It aggregates a base single model registry from :data:`aiqclib.common.loader.single_model_registry.SINGLE_MODEL_REGISTRY`
and extends it with specific suite models, offering convenient aliases.
Each key in the dictionary corresponds to a model name (string), and each value
is the class constructor for that model, typically inheriting from
:class:`aiqclib.common.base.model_base.ModelBase`.
"""

from typing import Dict, Type

from aiqclib.common.base.model_base import ModelBase
from aiqclib.common.loader.single_model_registry import SINGLE_MODEL_REGISTRY
from aiqclib.train.models.model_suite import ModelSuite

#: A dictionary mapping model names to their corresponding Python classes.
#:
#: This registry is initialized with models from :data:`~aiqclib.common.loader.single_model_registry.SINGLE_MODEL_REGISTRY`
#: and then updated to include the :class:`~aiqclib.train.models.model_suite.ModelSuite`
#: under both "ModelSuite" and "MS" keys, providing convenient aliases.
#:
#: The keys are strings (e.g., "XGBoost", "ModelSuite"), and the values are
#: class objects that inherit from :class:`~aiqclib.common.base.model_base.ModelBase`.
#:
#: :type: Dict[str, Type[ModelBase]]
MODEL_REGISTRY: Dict[str, Type[ModelBase]] = SINGLE_MODEL_REGISTRY
MODEL_REGISTRY.update({"ModelSuite": ModelSuite, "MS": ModelSuite})
