"""Unit tests for the ``ConfigBase`` class.

Coverage:
- Direct instantiation of the abstract ``ConfigBase`` raises NotImplementedError
- Invalid section name raises ValueError
- ``__repr__`` returns a one-line representation naming the concrete class,
  the section and the selected entry
- ``__str__`` / ``summary()`` describe the resolved configuration, and the
  step directories they report match the paths the pipeline actually uses
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

import os

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
        """__repr__ names the concrete class, the section and the selection."""
        ds = ConfigBaseWithExpectedName("data_sets", dataset_yaml_001)
        assert repr(ds) == (
            "ConfigBaseWithExpectedName(section_name=data_sets, dataset_name=None)"
        )

        ds.select("NRT_BO_001")
        assert "dataset_name=NRT_BO_001" in repr(ds)

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
    (DataSetConfig, "template:data_sets", "dataset_0001"),
    (DataSetConfig, "template:data_sets_full", "dataset_0001"),
    (TrainingConfig, "template:training_sets", "training_0001"),
    (ClassificationConfig, "template:classification_sets_full", "classification_0001"),
    (ClassificationConfig, "template:classification_sets", "classification_0001"),
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


# ---------------------------------------------------------------------------
# Tests for the printable summary (``print(config)``)
# ---------------------------------------------------------------------------


class TestConfigSummary:
    """Tests for ``summary()`` and the ``__str__`` it backs.

    The summary is documentation of a resolved configuration, so the tests
    check that it reports what the pipeline will actually do — in particular
    that each step's directory is the one the step class resolves for itself,
    which is not a given: ``input`` and (in classification) ``model`` are read
    without the dataset folder that every other step includes.
    """

    def test_str_is_the_summary(self, dataset_config_001):
        """``print(config)`` shows the summary; ``repr`` stays a single line."""
        assert str(dataset_config_001) == dataset_config_001.summary()
        assert "\n" in str(dataset_config_001)
        assert "\n" not in repr(dataset_config_001)

    def test_summary_before_select(self, dataset_yaml_001):
        """An unselected config lists the entries instead of resolving one."""
        conf = DataSetConfig(str(dataset_yaml_001))
        text = conf.summary()

        assert "<nothing selected>" in text
        assert "NRT_BO_001" in text
        assert "select(" in text
        # Nothing is resolved yet, so no step table is printed.
        assert "steps" not in text

    def test_summary_reports_targets_and_features(self, dataset_config_001):
        """Targets carry their flag column and flag values; features are listed."""
        text = dataset_config_001.summary()

        assert "temp (flag temp_qc, pos [4], neg [1])" in text
        for feature in dataset_config_001.data["feature_set"]["features"]:
            assert feature in text

    def test_summary_step_directories_match_the_pipeline(self, dataset_config_001):
        """Every step's reported directory is the one its files land in.

        ``get_full_file_name`` is what the step classes call, so comparing
        against it catches a summary that composes the path differently —
        the ``input`` step, which resolves without the dataset folder, is the
        case that would otherwise slip through.
        """
        text = dataset_config_001.summary()

        for step_name in dataset_config_001.data["step_class_set"]["steps"]:
            expected = os.path.dirname(
                dataset_config_001.get_full_file_name(
                    step_name,
                    default_file_name="file.parquet",
                    use_dataset_folder=step_name != "input",
                )
            )
            assert expected in text

    def test_summary_reports_row_filters(self, classify_config_001):
        """Active row filters are surfaced, since an empty result is silent."""
        assert "keep_years [2023]" in classify_config_001.summary()

    def test_summary_flags_unapplied_filters(self, classify_config_001):
        """A filter that is configured but switched off says so."""
        classify_config_001.data["step_param_set"]["steps"]["input"]["sub_steps"][
            "filter_rows"
        ] = False
        assert "not applied" in classify_config_001.summary()

    def test_summary_omits_absent_sections(self, training_config_001):
        """A training config has no features or input file, so neither is shown."""
        text = training_config_001.summary()

        assert "features" not in text
        assert "\n  input " not in text
        assert "KFoldValidation" in text

    def test_summary_lists_qc_items(self, nrtqc_config_001):
        """An NRT QC config reports its enabled items where features would be."""
        text = nrtqc_config_001.summary()

        assert "qc items" in text
        for item_name in nrtqc_config_001.get_qc_item_names():
            assert item_name in text

    def test_summary_reports_an_invalid_schema(self, config_dir):
        """A config that fails validation says so instead of hiding it."""
        conf = DataSetConfig(str(config_dir / "test_dataset_invalid.yaml"))
        text = conf.summary()

        assert "invalid" in text
        # The file holds no usable entries, so it says so rather than
        # inviting a select() that cannot succeed.
        assert "no 'data_sets' entries" in text

    def test_summary_does_not_validate_in_place(self, config_dir):
        """Printing a config leaves ``valid_yaml`` alone.

        ``summary()`` reports the schema status, but callers branch on
        ``valid_yaml``, so it must not be set as a side effect of printing.
        """
        conf = DataSetConfig(str(config_dir / "test_dataset_001.yaml"))
        assert conf.valid_yaml is False

        conf.summary()
        assert conf.valid_yaml is False

        assert conf.check_schema()[0] is True
        assert conf.valid_yaml is False

    def test_summary_survives_a_broken_path_info(self, dataset_config_001):
        """An unresolvable path is reported, not raised.

        Printing a configuration to work out why it is wrong must not fail on
        the very thing that is wrong.
        """
        dataset_config_001.data["path_info"]["common"]["base_path"] = None
        assert "<unresolved>" in dataset_config_001.summary()
