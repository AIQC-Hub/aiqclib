"""
This module provides the `ModelBase` abstract base class, which serves as the foundational
interface for all machine learning model implementations within the library. It enforces
a consistent structure for building, testing, and persisting models while managing
configuration and result storage.
"""

import os
from abc import ABC, abstractmethod
from typing import Optional, Any, Self

import polars as pl
from joblib import dump, load

from aiqclib.common.base.config_base import ConfigBase


class ModelBase(ABC):
    """
    Abstract base class for modeling tasks.

    Subclasses must define:

    - ``expected_class_name`` to match the configuration.
    - The :meth:`build` method for model building.
    - The :meth:`test` method for model testing.

    .. note::

       Since this class inherits from :class:`abc.ABC`, it cannot be directly
       instantiated and must be subclassed.
    """

    expected_class_name: Optional[str] = None  # Must be overridden by child classes
    short_name: Optional[str] = None  # Must be overridden by child classes
    multi = False  # Must be set to True for model suite class

    def __init__(self, config: ConfigBase) -> None:
        """
        Initialize the model with configuration data and validate
        that the expected class name matches what's in the YAML configuration.

        :param config: A configuration object providing parameters needed for model assembly and execution.
        :type config: ConfigBase
        :raises NotImplementedError: If ``expected_class_name`` is not defined in a subclass.
        :raises ValueError: If the class name derived from the configuration does not match the
                            ``expected_class_name`` or ``short_name`` of this class.
        """
        if not self.expected_class_name:
            raise NotImplementedError(
                "Child class must define 'expected_class_name' attribute"
            )

        # Validate that the YAML's "class" matches the child's declared class name
        base_class = config.get_base_class("model")
        if (base_class != self.expected_class_name) and (base_class != self.short_name):
            raise ValueError(
                f"Configuration mismatch: expected class '{self.expected_class_name}' "
                f"but got '{base_class}'"
            )

        model_params = config.data["step_param_set"]["steps"]["model"].get(
            "model_params", {}
        )

        self.config: ConfigBase = config
        self.model_params: dict = model_params

        self.training_set: Optional[Any] = None
        self.test_set: Optional[Any] = None
        self.model: Optional[Any] = None
        self.predictions: Optional[Any] = None
        self.report: Optional[Any] = None
        self.contingency_table: Optional[pl.DataFrame] = None
        self.k: int = 0
        self.allow_na = True

        # Check config to see if SHAP should be calculated
        self.enable_shap: bool = self.config.get_step_params("model").get(
            "calculate_shap", False
        )

        # Initialize storage for SHAP values explicitly
        self.shap_values: Optional[pl.DataFrame] = None

    @abstractmethod
    def build(self) -> None:
        """
        Build the model architecture or pipeline.

        Subclasses must implement logic to create, configure, and compile the model.
        """
        pass  # pragma: no cover

    @abstractmethod
    def test(self) -> None:
        """
        Evaluate the model performance on a provided test set or validation data.

        Subclasses must implement how the model is used to make predictions
        and how accuracy or performance measures are computed.
        """
        pass  # pragma: no cover

    @abstractmethod
    def update_nthreads(self, model: Self) -> Self:
        """
        Update the number of threads set in the model.

        Subclasses must implement logic to update the number of threads.

        :param model: The model instance that needs to be updated.
        :type model: Self
        :return: The model instance with updated thread settings.
        :rtype: Self
        """
        pass  # pragma: no cover

    @abstractmethod
    def _get_model_class(self) -> Any:
        """
        Return the class type of the underlying model to be instantiated.

        :return: The class object (e.g., xgboost.XGBClassifier, sklearn.linear_model.LogisticRegression).
        :rtype: Any
        """
        pass

    def load_model(self, file_name: str) -> None:
        """
        Load or deserialize a model from the given file path.

        :param file_name: The path to the file from which the model will be loaded.
        :type file_name: str
        :raises FileNotFoundError: If the specified file does not exist.
        :raises ValueError: If the loaded model type does not match the expected class
                            defined by the configuration.
        """
        if not os.path.exists(file_name):
            raise FileNotFoundError(f"File '{file_name}' does not exist.")

        self.model = load(file_name)
        expected_class = self._get_model_class()

        if not isinstance(self.model, expected_class):
            raise ValueError(
                f"Inconsistent class instances between config entry and loaded model. "
                f"Expected '{expected_class.__name__}', but got '{type(self.model).__name__}'."
            )

        if not isinstance(self.model, self._get_model_class()):
            raise ValueError(
                "Inconsistent class instances between config entry and loaded model."
            )

    def save_model(self, file_name: str) -> None:
        """
        Save or serialize the current model to the provided file path.

        :param file_name: The path indicating where the model will be saved.
        :type file_name: str
        """
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        dump(self.model, file_name)

    def update_contingency_table(self) -> None:
        """
        Updates the internal contingency table with the current test set predictions.

        This method extracts the fold index (`k`), ground truth (`label`), and
        predicted probability (`score`) from the current test set and predictions.
        The data is stored in the :attr:`contingency_table` attribute as a Polars DataFrame.

        If :attr:`contingency_table` is already populated (e.g., during cross-validation),
        the new results are appended (vstacked) to the existing DataFrame.

        :raises ValueError: If :attr:`test_set` or :attr:`predictions` are ``None``.
        """
        if self.test_set is None:
            raise ValueError("Member variable 'test_set' must not be empty.")

        if self.predictions is None:
            raise ValueError("Member variable 'predictions' must not be empty.")

        # Create a DataFrame for the current fold/batch
        current_data = pl.DataFrame(
            {
                "k": self.k,
                "label": self.test_set["label"],
                "predicted_label": self.predictions["predicted_label"],
                "score": self.predictions["score"],
            }
        )

        # Append to the existing table if it exists, otherwise initialize it
        if self.contingency_table is None:
            self.contingency_table = current_data
        else:
            self.contingency_table = self.contingency_table.vstack(current_data)

    def __repr__(self) -> str:
        """
        Return a string representation of the ModelBase instance.

        :return: A string describing the instance with its class name declared by ``expected_class_name``.
        :rtype: str
        """
        return f"ModelBase(class={self.expected_class_name})"
