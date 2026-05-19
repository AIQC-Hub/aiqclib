"""Unit tests for the ``DataSetConfig`` class.

Coverage:
- ``validate()`` correctly identifies valid vs. invalid YAML files
- ``select()`` populates the expected ds.data sections with the right
  number of keys for a prepare-stage config
- Repeated ``select()`` calls are idempotent
- Selecting an unknown name raises ValueError
- Template YAMLs (all three variants) resolve folder paths correctly for
  input/summary/split
- ``auto_select=False`` defers data loading

Refactored from a ``unittest.TestCase`` + a pytest-style template class
into two parallel pytest classes. Both classes get their YAML paths from
conftest fixtures (``dataset_yaml_001``, ``config_dir``) rather than
constructing paths inline.
"""

import pytest

from aiqclib.common.config.dataset_config import DataSetConfig


# ---------------------------------------------------------------------------
# Basic config tests against test_dataset_001.yaml
# ---------------------------------------------------------------------------

class TestDataSetConfig:
    """Tests for ``DataSetConfig`` against a real test YAML."""

    def test_valid_config(self, dataset_yaml_001):
        """A well-formed YAML validates as 'valid'."""
        ds = DataSetConfig(str(dataset_yaml_001))
        assert "valid" in ds.validate()

    def test_invalid_config(self, config_dir):
        """A malformed YAML validates as 'invalid'.

        Relies on ``test_dataset_invalid.yaml`` in the config fixtures.
        """
        ds = DataSetConfig(str(config_dir / "test_dataset_invalid.yaml"))
        assert "invalid" in ds.validate()

    def test_load_dataset_config(self, dataset_yaml_001):
        """After select(), each top-level data section has the expected key count.

        Prepare-stage configs have 6 path_info keys (vs 8 for classification,
        which adds `model` and `concat`).
        """
        ds = DataSetConfig(str(dataset_yaml_001))
        ds.select("NRT_BO_001")

        assert len(ds.data["path_info"]) == 6
        assert len(ds.data["target_set"]) == 2
        assert len(ds.data["feature_set"]) == 2
        assert len(ds.data["feature_param_set"]) == 2
        assert len(ds.data["step_class_set"]) == 2
        assert len(ds.data["step_param_set"]) == 2

    def test_load_dataset_config_twice(self, dataset_yaml_001):
        """Calling select() twice with the same name is idempotent."""
        ds = DataSetConfig(str(dataset_yaml_001))
        ds.select("NRT_BO_001")
        ds.select("NRT_BO_001")  # Should not raise

    def test_invalid_dataset_name(self, dataset_yaml_001):
        """select() with an unknown dataset name raises ValueError."""
        ds = DataSetConfig(str(dataset_yaml_001))
        with pytest.raises(ValueError):
            ds.select("INVALID_NAME")


# ---------------------------------------------------------------------------
# Template-config tests against the in-package YAML templates
# ---------------------------------------------------------------------------

# Module-level list of template files. All three templates resolve the same
# folder paths; tests parametrize over idx. Note that ``test_input_folder``
# only iterates over the first two — the third uses a different default
# input file name, preserved from the original test.
_TEMPLATE_FILENAMES = (
    "config_data_set_full_template.yaml",
    "config_data_set_reduced_template.yaml",
    "config_data_set_template.yaml",
)


class TestDataSetConfigTemplate:
    """Tests for the bundled prepare-stage template YAMLs."""

    @pytest.fixture
    def template_paths(self, config_dir):
        """List of full paths to the dataset template YAMLs."""
        return [config_dir / name for name in _TEMPLATE_FILENAMES]

    @pytest.mark.parametrize("idx", range(2))
    def test_input_folder(self, idx, template_paths):
        """Input file path resolves correctly under the first two templates.

        Original only parametrized over range(2) — the third template uses
        a different input_file_name. Preserving that scope.
        """
        ds = DataSetConfig(str(template_paths[idx]))
        ds.select("dataset_0001")
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
        ds = DataSetConfig(str(template_paths[idx]))
        ds.select("dataset_0001")
        assert (
            ds.get_full_file_name("summary", "test.txt")
            == "/path/to/data/dataset_0001/summary/test.txt"
        )

    @pytest.mark.parametrize("idx", range(len(_TEMPLATE_FILENAMES)))
    def test_split_folder(self, idx, template_paths):
        """Files placed in 'split' resolve under the 'training' folder.

        Note: split is mapped to the 'training' physical subfolder by the
        default template path_info — these names are decoupled by design.
        """
        ds = DataSetConfig(str(template_paths[idx]))
        ds.select("dataset_0001")
        assert (
            ds.get_full_file_name("split", "test.txt")
            == "/path/to/data/dataset_0001/training/test.txt"
        )

    @pytest.mark.parametrize("idx", range(len(_TEMPLATE_FILENAMES)))
    def test_auto_select(self, idx, template_paths):
        """auto_select=False defers loading; auto_select=True loads immediately."""
        ds = DataSetConfig(str(template_paths[idx]), False)
        assert ds.data is None

        ds = DataSetConfig(str(template_paths[idx]), True)
        assert ds.data is not None