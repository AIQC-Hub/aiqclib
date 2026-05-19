"""Unit tests for the ``ConfigBase`` class.

Coverage:
- Direct instantiation of the abstract ``ConfigBase`` raises NotImplementedError
- Invalid section name raises ValueError
- ``__str__`` returns a structured representation including section name
- A corrupted ``full_config`` causes ``select()`` to raise ValueError
- Missing ``base_path`` in the ``common`` section causes ``get_base_path()``
  to raise ValueError
- All five bundled template YAMLs (data_sets, data_sets_full, training_sets,
  classification_sets, classification_sets_full) load and select correctly
  via the corresponding config class

Refactored from a ``unittest.TestCase`` + a pytest-style template class.
The mock subclass ``ConfigBaseWithExpectedName`` stays at module level.
Setup is replaced by the ``dataset_yaml_001`` fixture — ConfigBase loads
its own YAML, so the tests need the *path*, not a pre-loaded config.

Class rename: the first class was named ``TestDatasetBaseMethods`` in the
original — a copy-paste typo, since it tests ``ConfigBase`` (not
``DataSetBase``, which lives in ``test_common_base_dataset.py``). Renamed
to ``TestConfigBaseMethods`` so the name matches the class under test.
``pytest -k`` filters now target the correct test surface.
"""

import pytest

from aiqclib.common.base.config_base import ConfigBase
from aiqclib.common.config.classify_config import ClassificationConfig
from aiqclib.common.config.dataset_config import DataSetConfig
from aiqclib.common.config.training_config import TrainingConfig


# ---------------------------------------------------------------------------
# Module-level mock subclass
# ---------------------------------------------------------------------------

class ConfigBaseWithExpectedName(ConfigBase):
    """Minimal concrete subclass for exercising ConfigBase plumbing."""

    expected_class_name: str = "ConfigBaseWithExpectedName"

    def __init__(self, section_name: str, config_file: str) -> None:
        super().__init__(section_name, config_file)


# ---------------------------------------------------------------------------
# Tests for ConfigBase methods (renamed from TestDatasetBaseMethods)
# ---------------------------------------------------------------------------

class TestConfigBaseMethods:
    """Tests for ConfigBase's abstract-class behaviour, __str__, and validation.

    Renamed from ``TestDatasetBaseMethods`` — the original was a copy-paste
    from ``test_common_base_dataset.py``. This file tests ConfigBase.
    """

    def test_common_base_path(self, dataset_yaml_001):
        """Direct instantiation of ConfigBase raises NotImplementedError.

        ConfigBase is abstract — subclasses must define
        ``expected_class_name``.
        """
        with pytest.raises(NotImplementedError):
            _ = ConfigBase("data_sets", dataset_yaml_001)

    def test_section_name(self, dataset_yaml_001):
        """An unsupported section name raises ValueError."""
        with pytest.raises(ValueError):
            _ = ConfigBaseWithExpectedName("invalid_section_name", dataset_yaml_001)

    def test_represented_str(self, dataset_yaml_001):
        """__str__ returns "ConfigBase(section_name=<section>)"."""
        ds = ConfigBaseWithExpectedName("data_sets", dataset_yaml_001)
        assert str(ds) == "ConfigBase(section_name=data_sets)"

    def test_validation_error_with_select(self, dataset_yaml_001):
        """select() on a corrupted full_config raises ValueError.

        Manually corrupts ``ds.full_config`` to simulate an invalid YAML
        structure and verifies select() catches the schema violation.
        """
        ds = ConfigBaseWithExpectedName("data_sets", dataset_yaml_001)
        ds.full_config = ""
        with pytest.raises(ValueError):
            ds.select("NRT_BO_001")

    def test_no_base_name(self, dataset_yaml_001):
        """get_base_path() raises ValueError when common.base_path is None."""
        ds = ConfigBaseWithExpectedName("data_sets", dataset_yaml_001)
        ds.select("NRT_BO_001")
        ds.data["path_info"]["common"]["base_path"] = None
        with pytest.raises(ValueError):
            ds.get_base_path("invalid_step_name")


# ---------------------------------------------------------------------------
# Tests for the bundled template YAMLs (each loadable via its config class)
# ---------------------------------------------------------------------------

# (config_class, template_path, select_name) tuples covering all five
# bundled templates. The ``template:`` prefix triggers in-package template
# resolution rather than disk loading.
_TEMPLATE_CASES = [
    (DataSetConfig,        "template:data_sets",                  "dataset_0001"),
    (DataSetConfig,        "template:data_sets_full",             "dataset_0001"),
    (TrainingConfig,       "template:training_sets",              "training_0001"),
    (ClassificationConfig, "template:classification_sets_full",   "classification_0001"),
    (ClassificationConfig, "template:classification_sets",        "classification_0001"),
]


class TestConfigTemplates:
    """Tests for loading the five bundled YAML templates via each config class."""

    @pytest.mark.parametrize(
        "config_class, template_path, select_name",
        _TEMPLATE_CASES,
        ids=[f"{cls.__name__}:{path}" for cls, path, _ in _TEMPLATE_CASES],
    )
    def test_read_template(self, config_class, template_path, select_name):
        """Each template loads, then select() populates ``data``.

        Before select(), ``data`` is None (auto_select is False by default
        for templates). After select(), ``data`` is populated with the
        selected dataset/training/classification entry.
        """
        conf = config_class(template_path)
        assert conf.full_config is not None
        assert conf.data is None

        conf.select(select_name)
        assert conf.data is not None