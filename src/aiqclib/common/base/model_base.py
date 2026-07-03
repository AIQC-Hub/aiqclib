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
        self.model_score: Optional[pl.DataFrame] = None
        self.k: int = 0
        self.allow_na = True

        # When True, :meth:`test` runs prediction only and skips performance
        # evaluation (report + model scores). Defaults to False so the training
        # pipeline is unaffected; the classification pipeline sets it per target
        # via ``config.get_skip_evaluation`` for label-free datasets.
        self.skip_evaluation: bool = False

        # Check config to see if SHAP should be calculated
        self.enable_shap: bool = self.config.get_step_params("model").get(
            "calculate_shap", False
        )

        # Score threshold used to convert predicted probabilities into binary
        # labels: predicted_label = 1 if score >= threshold else 0. Defaults to
        # 0.5 when not configured (backward-compatible with the previous
        # hardcoded behaviour). Stored as an instance attribute so it is
        # serialized with the model and recovered on load.
        self.predicted_label_threshold: float = self.config.get_step_params(
            "model"
        ).get("predicted_label_threshold", 0.5)

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

    def update_model_score(self) -> None:
        """
        Updates the internal model-scores table with the current test set predictions.

        Each row records the model that produced the prediction (`method`), the
        fold index (`k`), the ground truth (`label`), and the predicted
        probability (`score`). The data is stored in the :attr:`model_score`
        attribute as a Polars DataFrame.

        The ``method`` column is the lowercased ``short_name`` of the model
        (e.g. ``"xgb"``, ``"dt"``) and is always present, for both single-model
        and suite pipelines. This makes the model-scores file self-describing
        about which model produced each row.

        Note that ``predicted_label`` is intentionally NOT stored: it is
        derivable from ``score`` and a threshold (``score >= threshold``), so
        keeping it would bake in a single threshold and make the file less
        useful for external threshold-sweeping (ROC/PR analysis). Consumers
        apply their own threshold to ``score`` as needed.

        If :attr:`model_score` is already populated (e.g., during cross-validation),
        the new results are appended (vstacked) to the existing DataFrame.

        :raises ValueError: If :attr:`test_set` or :attr:`predictions` are ``None``.
        """
        if self.test_set is None:
            raise ValueError("Member variable 'test_set' must not be empty.")

        if self.predictions is None:
            raise ValueError("Member variable 'predictions' must not be empty.")

        method = getattr(self, "short_name", "").lower()

        # Create a DataFrame for the current fold/batch.
        # Column order: method, k, label, score.
        #
        # Normalize dtypes so that frames from different models concat cleanly
        # in the suite path: XGBoost's predict_proba yields Float32 while other
        # sklearn models yield Float64, so cast score to Float64. k comes from a
        # Python int but is cast to Int64 explicitly for cross-fold/cross-method
        # consistency. (The old suite code did these casts per-method; doing it
        # here makes every model_score frame uniform at the source.)
        current_data = pl.DataFrame(
            {
                "method": method,
                "k": self.k,
                "label": self.test_set["label"],
                "score": self.predictions["score"],
            }
        ).with_columns(
            [
                pl.col("k").cast(pl.Int64),
                pl.col("score").cast(pl.Float64),
            ]
        )

        # Append to the existing table if it exists, otherwise initialize it
        if self.model_score is None:
            self.model_score = current_data
        else:
            self.model_score = self.model_score.vstack(current_data)

    def __repr__(self) -> str:
        """
        Return a string representation of the ModelBase instance.

        :return: A string describing the instance with its class name declared by ``expected_class_name``.
        :rtype: str
        """
        return f"ModelBase(class={self.expected_class_name})"
