"""Unit tests for the ``DataSetBase`` class.

Coverage:
- Direct instantiation of the abstract ``DataSetBase`` raises
  NotImplementedError
- Instantiating a concrete subclass with a step name that doesn't match
  the configured step class raises ValueError
- ``__str__`` returns a structured representation including step name and
  class name

Refactored from a ``unittest.TestCase`` class. The mock subclass
``DataSetWithExpectedName`` stays at module level (test infrastructure,
not data). The setUp is replaced by the ``dataset_config_001`` fixture
from conftest.
"""

import pytest

from aiqclib.common.base.config_base import ConfigBase
from aiqclib.common.base.dataset_base import DataSetBase


# ---------------------------------------------------------------------------
# Module-level mock subclass
# ---------------------------------------------------------------------------


class DataSetWithExpectedName(DataSetBase):
    """Minimal concrete subclass for exercising DataSetBase plumbing."""

    expected_class_name: str = "InputDataSetA"

    def __init__(self, step_name: str, config: ConfigBase) -> None:
        super().__init__(step_name, config)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDatasetBaseMethods:
    """Tests for DataSetBase's abstract-class behaviour and __str__."""

    def test_common_base_path(self, dataset_config_001):
        """Direct instantiation of DataSetBase raises NotImplementedError.

        DataSetBase is abstract — subclasses must define
        ``expected_class_name``. Constructing it directly should fail before
        the constructor reaches any data validation.
        """
        with pytest.raises(NotImplementedError):
            _ = DataSetBase("input", dataset_config_001)

    def test_step_name(self, dataset_config_001):
        """Mismatched step name raises ValueError.

        ``DataSetWithExpectedName`` declares ``expected_class_name = "InputDataSetA"``,
        and the config's step_class_set has ``InputDataSetA`` registered under
        the ``input`` step. Constructing it with step_name=``select`` should
        therefore raise.
        """
        with pytest.raises(ValueError):
            _ = DataSetWithExpectedName("select", dataset_config_001)

    def test_represented_str(self, dataset_config_001):
        """__str__ returns "DataSetBase(step=<step>, class=<expected_class>)"."""
        ds = DataSetWithExpectedName("input", dataset_config_001)
        assert str(ds) == "DataSetBase(step=input, class=InputDataSetA)"
