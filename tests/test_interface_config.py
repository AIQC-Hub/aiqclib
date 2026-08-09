"""Unit tests for the public config interface: ``read_config`` and ``write_config_template``.

Coverage:
- write_config_template produces a file at the requested path for each
  (module, variant) combination; invalid module/path inputs raise.
- read_config returns the appropriate config class (DataSetConfig,
  TrainingConfig, ClassificationConfig) based on the file's contents;
  invalid module/path inputs raise.

Refactored from two pytest classes (already pytest, not unittest) into the
same structure with conftest fixtures for paths. The template-list data is
hoisted to a module-level constant since the test bodies index into it by
the parametrized ``idx``.
"""

import os

import pytest

from aiqclib.common.config.classify_config import ClassificationConfig
from aiqclib.common.config.dataset_config import DataSetConfig
from aiqclib.common.config.training_config import TrainingConfig
from aiqclib.interface.config import read_config, write_config_template


# (module, variant, filename) triples for the write_config_template parametrize.
# Variant "" means the default template for that module.
TEMPLATE_SPECS = [
    ("prepare", "", "temp_dataset_template.yaml"),
    ("prepare", "full", "temp_dataset_full_template.yaml"),
    ("prepare", "reduced", "temp_dataset_reduced_template.yaml"),
    ("train", "", "temp_training_template.yaml"),
    ("classify", "", "temp_classification_template.yaml"),
    ("classify", "full", "temp_classification_template.yaml"),
]


class TestTemplateConfig:
    """Tests for ``write_config_template``."""

    @pytest.mark.parametrize("idx", range(len(TEMPLATE_SPECS)))
    def test_write_config_template(self, idx, test_output_dir):
        """Each (module, variant) writes a template file to the requested path."""
        module, variant, filename = TEMPLATE_SPECS[idx]
        path = test_output_dir / filename
        write_config_template(str(path), module, variant)
        assert os.path.exists(path)
        os.remove(path)  # comment out to debug

    def test_config_template_with_invalid_module(self, test_output_dir):
        """An unknown module name raises ValueError."""
        with pytest.raises(ValueError):
            write_config_template(
                str(test_output_dir / "temp_dataset_template.yaml"),
                "prepare2",
            )

    def test_config_template_with_invalid_path(self):
        """An unwritable path (under a non-existent root) raises IOError."""
        with pytest.raises(IOError):
            write_config_template("/abc/temp_dataset_template.yaml", "prepare")

    def test_missing_directory_message_names_the_option(self, tmp_path):
        """The refusal tells the user how to create the directory instead."""
        path = tmp_path / "not_yet" / "prepare.yaml"
        with pytest.raises(IOError, match="create_dirs=True"):
            write_config_template(str(path), "prepare")
        assert not path.parent.exists()

    def test_create_dirs_creates_missing_directories(self, tmp_path):
        """With create_dirs=True the whole missing path is created."""
        path = tmp_path / "a" / "b" / "c" / "prepare.yaml"
        write_config_template(str(path), "prepare", create_dirs=True)
        assert path.exists()

    def test_create_dirs_accepts_an_existing_directory(self, tmp_path):
        """create_dirs=True is a no-op when the directory is already there."""
        path = tmp_path / "prepare.yaml"
        write_config_template(str(path), "prepare", create_dirs=True)
        assert path.exists()

    def test_bare_filename_needs_no_directory(self, tmp_path, monkeypatch):
        """A filename with no directory part is written to the working directory."""
        monkeypatch.chdir(tmp_path)
        write_config_template("prepare.yaml", "prepare")
        assert (tmp_path / "prepare.yaml").exists()

    def test_tilde_path_is_not_expanded(self, tmp_path):
        """A '~' path is not expanded, and the message says so."""
        with pytest.raises(IOError, match="not expanded automatically"):
            write_config_template("~/aiqclib_no_such_dir/prepare.yaml", "prepare")


class TestReadConfig:
    """Tests for ``read_config``."""

    @pytest.mark.parametrize(
        "config_fixture_name",
        ["dataset_yaml_001", "dataset_yaml_004"],
    )
    def test_ds_config(self, config_fixture_name, request):
        """Reading a dataset config file returns a DataSetConfig instance."""
        path = request.getfixturevalue(config_fixture_name)
        config = read_config(path)
        assert isinstance(config, DataSetConfig)

    def test_train_config(self, training_yaml_001):
        """Reading a training config file returns a TrainingConfig instance."""
        config = read_config(training_yaml_001, "NRT_BO_001", False)
        assert isinstance(config, TrainingConfig)

    def test_train_config_with_multiple_entries(self, training_yaml_001):
        """Reading a training config file returns a TrainingConfig instance."""
        with pytest.raises(ValueError):
            _ = read_config(training_yaml_001, "NRT_BO_001")

    @pytest.mark.parametrize(
        "config_fixture_name",
        ["classify_yaml_001", "classify_yaml_002"],
    )
    def test_classify_config(self, config_fixture_name, request):
        """Reading a classify config file returns a ClassificationConfig instance."""
        path = request.getfixturevalue(config_fixture_name)
        config = read_config(path)
        assert isinstance(config, ClassificationConfig)

    def test_config_with_invalid_module(self, config_dir):
        """A YAML file with an unrecognized config_type raises ValueError.

        Relies on tests/data/config/test_dataset_invalid.yaml being a fixture
        file with a bad ``config_type`` field.
        """
        with pytest.raises(ValueError):
            _ = read_config(config_dir / "test_dataset_invalid.yaml")

    def test_config_with_invalid_path(self, dataset_yaml_001):
        """A non-existent file path raises IOError."""
        with pytest.raises(IOError):
            _ = read_config(str(dataset_yaml_001) + "abc")
