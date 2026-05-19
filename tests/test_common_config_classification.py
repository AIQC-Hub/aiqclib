"""Unit tests for the ``ClassificationConfig`` class.

Coverage:
- ``validate()`` correctly identifies valid vs. invalid YAML files
- ``select()`` populates the expected ds.data sections with the right
  number of keys for a classification config
- Repeated ``select()`` calls are idempotent
- Selecting an unknown name raises ValueError
- Template YAMLs resolve folder paths correctly for input/summary/classify
- ``auto_select=False`` defers data loading

Refactored from a ``unittest.TestCase`` + a pytest-style template class
into two parallel pytest classes. Both classes get their YAML paths from
conftest fixtures (``classify_yaml_001``, ``config_dir``) rather than
constructing paths inline.
"""

import pytest

from aiqclib.common.config.classify_config import ClassificationConfig


# ---------------------------------------------------------------------------
# Basic config tests against test_classify_001.yaml
# ---------------------------------------------------------------------------


class TestClassificationConfig:
    """Tests for ``ClassificationConfig`` against a real test YAML."""

    def test_valid_config(self, classify_yaml_001):
        """A well-formed YAML validates as 'valid'."""
        ds = ClassificationConfig(str(classify_yaml_001))
        assert "valid" in ds.validate()

    def test_invalid_config(self, config_dir):
        """A malformed YAML validates as 'invalid'.

        Relies on ``test_dataset_invalid.yaml`` in the config fixtures.
        """
        ds = ClassificationConfig(str(config_dir / "test_dataset_invalid.yaml"))
        assert "invalid" in ds.validate()

    def test_load_dataset_config(self, classify_yaml_001):
        """After select(), each top-level data section has the expected key count.

        Classification configs include `model` and `concat` path entries, so
        path_info has 8 keys (vs 6 for prepare/training configs). The other
        sections each have 2 keys (`name` + body).
        """
        ds = ClassificationConfig(str(classify_yaml_001))
        ds.select("NRT_BO_001")

        assert len(ds.data["path_info"]) == 8
        assert len(ds.data["target_set"]) == 2
        assert len(ds.data["feature_set"]) == 2
        assert len(ds.data["feature_param_set"]) == 2
        assert len(ds.data["step_class_set"]) == 2
        assert len(ds.data["step_param_set"]) == 2

    def test_load_dataset_config_twice(self, classify_yaml_001):
        """Calling select() twice with the same name is idempotent."""
        ds = ClassificationConfig(str(classify_yaml_001))
        ds.select("NRT_BO_001")
        ds.select("NRT_BO_001")  # Should not raise

    def test_invalid_dataset_name(self, classify_yaml_001):
        """select() with an unknown dataset name raises ValueError."""
        ds = ClassificationConfig(str(classify_yaml_001))
        with pytest.raises(ValueError):
            ds.select("INVALID_NAME")


# ---------------------------------------------------------------------------
# Template-config tests against the in-package YAML templates
# ---------------------------------------------------------------------------

# Module-level list of template files. Both templates resolve the same
# folder paths under the same select name; tests parametrize over idx.
_TEMPLATE_FILENAMES = (
    "config_classify_set_full_template.yaml",
    "config_classify_set_template.yaml",
)


class TestClassificationConfigTemplate:
    """Tests for the bundled classification template YAMLs."""

    @pytest.fixture
    def template_paths(self, config_dir):
        """List of full paths to the classification template YAMLs."""
        return [config_dir / name for name in _TEMPLATE_FILENAMES]

    @pytest.mark.parametrize("idx", range(len(_TEMPLATE_FILENAMES)))
    def test_input_folder(self, idx, template_paths):
        """Input file path resolves correctly under the template settings."""
        ds = ClassificationConfig(str(template_paths[idx]))
        ds.select("classification_0001")
        input_file_name = ds.get_full_file_name(
            "input",
            ds.data["input_file_name"],
            use_dataset_folder=False,
            folder_name_auto=False,
        )
        assert input_file_name == "/path/to/input/nrt_cora_bo_4.parquet"

    @pytest.mark.parametrize("idx", range(len(_TEMPLATE_FILENAMES)))
    def test_summary_folder(self, idx, template_paths):
        """Files placed in 'summary' resolve under the dataset folder."""
        ds = ClassificationConfig(str(template_paths[idx]))
        ds.select("classification_0001")
        assert (
            ds.get_full_file_name("summary", "test.txt")
            == "/path/to/data/dataset_0001/summary/test.txt"
        )

    @pytest.mark.parametrize("idx", range(len(_TEMPLATE_FILENAMES)))
    def test_classify_folder(self, idx, template_paths):
        """Files placed in 'classify' resolve under the dataset folder."""
        ds = ClassificationConfig(str(template_paths[idx]))
        ds.select("classification_0001")
        assert (
            ds.get_full_file_name("classify", "test.txt")
            == "/path/to/data/dataset_0001/classify/test.txt"
        )

    @pytest.mark.parametrize("idx", range(len(_TEMPLATE_FILENAMES)))
    def test_auto_select(self, idx, template_paths):
        """auto_select=False defers loading; auto_select=True loads immediately."""
        ds = ClassificationConfig(str(template_paths[idx]), False)
        assert ds.data is None

        ds = ClassificationConfig(str(template_paths[idx]), True)
        assert ds.data is not None
