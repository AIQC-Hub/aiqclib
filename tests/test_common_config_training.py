"""Unit tests for the ``TrainingConfig`` class.

Coverage:
- ``validate()`` correctly identifies valid vs. invalid YAML files
- ``select()`` populates the expected ds.data sections with the right
  number of keys for a training config (no feature_set / feature_param_set,
  since training doesn't compute features)
- Selecting an unknown name raises ValueError
- The single training template YAML resolves folder paths correctly for
  input/valid/build
- ``auto_select=False`` defers data loading

Refactored from a single ``unittest.TestCase`` class. Structural change:
**split into two classes**, ``TestTrainingConfig`` (basic config tests)
and ``TestTrainingConfigTemplate`` (template path resolution + auto_select),
matching the parallel structure already used in ``test_common_config_dataset.py``
and ``test_common_config_classification.py``.

``pytest -k TestTrainingConfig`` still matches both classes, so existing
filters continue to work. The more specific
``pytest -k TestTrainingConfigTemplate`` now also works.
"""

import pytest

from aiqclib.common.config.training_config import TrainingConfig


# ---------------------------------------------------------------------------
# Basic config tests against test_training_001.yaml
# ---------------------------------------------------------------------------


class TestTrainingConfig:
    """Tests for ``TrainingConfig`` against a real test YAML."""

    def test_valid_config(self, training_yaml_001):
        """A well-formed YAML validates as 'valid'."""
        ds = TrainingConfig(str(training_yaml_001))
        assert "valid" in ds.validate()

    def test_invalid_config(self, config_dir):
        """A malformed YAML validates as 'invalid'.

        Relies on ``test_dataset_invalid.yaml`` in the config fixtures.
        """
        ds = TrainingConfig(str(config_dir / "test_dataset_invalid.yaml"))
        assert "invalid" in ds.validate()

    def test_load_dataset_config(self, training_yaml_001):
        """After select(), each top-level data section has the expected key count.

        Training configs don't have feature_set or feature_param_set
        sections — features are extracted at the prepare stage, not at
        training time.
        """
        ds = TrainingConfig(str(training_yaml_001))
        ds.select("NRT_BO_001")

        assert len(ds.data["path_info"]) == 6
        assert len(ds.data["target_set"]) == 2
        assert len(ds.data["step_class_set"]) == 2
        assert len(ds.data["step_param_set"]) == 2

    def test_invalid_dataset_name(self, training_yaml_001):
        """select() with an unknown dataset name raises ValueError."""
        ds = TrainingConfig(str(training_yaml_001))
        with pytest.raises(ValueError):
            ds.select("INVALID_NAME")


# ---------------------------------------------------------------------------
# Template-config tests against the bundled training template
# ---------------------------------------------------------------------------


class TestTrainingConfigTemplate:
    """Tests for the bundled training template YAML."""

    @pytest.fixture
    def template_path(self, config_dir):
        """Full path to the single training template YAML.

        Unlike the dataset/classification template tests (which parametrize
        over multiple template variants), training has just one template.
        """
        return config_dir / "config_train_set_template.yaml"

    def test_input_folder(self, template_path):
        """Files placed in 'input' resolve under the 'training' subfolder.

        Like prepare-stage's split→training mapping, training's 'input'
        physically lives under the 'training' folder — the logical step
        name and the physical folder name are decoupled by the template.
        """
        ds = TrainingConfig(str(template_path))
        ds.select("training_0001")
        assert (
            ds.get_full_file_name("input", "test.txt")
            == "/path/to/data/dataset_0001/training/test.txt"
        )

    def test_valid_folder(self, template_path):
        """Files placed in 'valid' resolve under the dataset folder."""
        ds = TrainingConfig(str(template_path))
        ds.select("training_0001")
        assert (
            ds.get_full_file_name("valid", "test.txt")
            == "/path/to/data/dataset_0001/valid/test.txt"
        )

    def test_build_folder(self, template_path):
        """Files placed in 'build' resolve under the dataset folder."""
        ds = TrainingConfig(str(template_path))
        ds.select("training_0001")
        assert (
            ds.get_full_file_name("build", "test.txt")
            == "/path/to/data/dataset_0001/build/test.txt"
        )

    def test_auto_select(self, template_path):
        """auto_select=False defers loading; auto_select=True loads immediately."""
        ds = TrainingConfig(str(template_path), False)
        assert ds.data is None

        ds = TrainingConfig(str(template_path), True)
        assert ds.data is not None
