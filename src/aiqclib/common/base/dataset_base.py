"""
This module defines the abstract base class `DataSetBase`, which serves as a
foundation for implementing various dataset classes.

It provides a common structure for dataset initialization, including validation
of the `expected_class_name` attribute against the provided configuration.
Subclasses are expected to override the `expected_class_name` attribute
to match their specific class identifier in the system's configuration.
"""

from abc import ABC

from aiqclib.common.base.config_base import ConfigBase


class DataSetBase(ABC):
    """
    Base class for dataset classes.

    Subclasses must define an ``expected_class_name`` attribute, which is used to
    validate the YAML entry's ``step_class_sets``.

    :ivar expected_class_name: The expected class name for validation against configuration.
                                This must be overridden by child classes.
    :vartype expected_class_name: str or None
    :ivar step_name: The name of the step identified in the configuration.
    :vartype step_name: str
    :ivar config: A configuration object that provides the necessary information.
    :vartype config: ConfigBase

    .. note::

       This class extends the :class:`abc.ABC` in order to indicate that it is
       an abstract base class.
    """

    expected_class_name: str | None = None  # Must be overridden by child classes

    def __init__(self, step_name: str, config: ConfigBase) -> None:
        """
        Initialize a new instance of DataSetBase.

        :param step_name: The name of the step identified in the configuration.
        :type step_name: str
        :param config: A configuration object that provides the necessary information
                       (e.g., the ``base_class`` entry).
        :type config: ConfigBase
        :raises NotImplementedError: If no ``expected_class_name`` is defined by a
                                     child class.
        :raises ValueError: If the YAML's ``base_class`` does not match the
                            ``expected_class_name``.
        """
        if not self.expected_class_name:
            raise NotImplementedError(
                "Child class must define 'expected_class_name' attribute"
            )

        # Validate that the YAML's "class" matches the child's declared class name
        base_class = config.get_base_class(step_name)
        if base_class != self.expected_class_name:
            raise ValueError(
                f"Configuration mismatch: expected class '{self.expected_class_name}' "
                f"but got '{base_class}'"
            )

        # Set member variables
        self.step_name: str = step_name
        self.config: ConfigBase = config

    def __repr__(self) -> str:
        """
        Return a string representation of the DataSetBase instance.

        :return: A string describing the instance with its ``step_name``
                 and the class name declared by ``expected_class_name``.
        :rtype: str
        """
        return f"DataSetBase(step={self.step_name}, class={self.expected_class_name})"
