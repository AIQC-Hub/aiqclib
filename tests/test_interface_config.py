"""Unit tests for the public config interface: ``read_config``,
``read_config_template`` and ``write_config_template``.

Coverage:
- write_config_template produces a file at the requested path for each
  (module, variant) combination; invalid module/path inputs raise.
- read_config returns the appropriate config class (DataSetConfig,
  TrainingConfig, ClassificationConfig) based on the file's contents;
  invalid module/path inputs raise.
- read_config_template returns the same template as its writing counterpart,
  as a resolved config object of the matching class.

Refactored from two pytest classes (already pytest, not unittest) into the
same structure with conftest fixtures for paths. The template-list data is
hoisted to a module-level constant since the test bodies index into it by
the parametrized ``idx``.
"""

import os

import pytest
import yaml

from aiqclib.common.config.classify_config import ClassificationConfig
from aiqclib.common.config.dataset_config import DataSetConfig
from aiqclib.common.config.nrtqc_config import NRTQCConfig
from aiqclib.common.config.training_config import TrainingConfig
from aiqclib.interface.config import (
    read_config,
    read_config_template,
    write_config_template,
)


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

    def test_existing_file_is_not_replaced_by_default(self, tmp_path):
        """A customized config must not be reset by regenerating the template."""
        path = tmp_path / "prepare.yaml"
        write_config_template(str(path), "prepare")
        path.write_text("# my edited config\n")

        with pytest.raises(FileExistsError, match="overwrite=True"):
            write_config_template(str(path), "prepare")
        assert path.read_text() == "# my edited config\n"

    def test_overwrite_replaces_the_file(self, tmp_path):
        """With the flag the template is written over the old contents."""
        path = tmp_path / "prepare.yaml"
        write_config_template(str(path), "prepare")
        path.write_text("# my edited config\n")

        write_config_template(str(path), "prepare", overwrite=True)
        assert "path_info_sets" in path.read_text()

    def test_overwrite_on_a_new_file_is_harmless(self, tmp_path):
        """The flag does not require the file to already exist."""
        path = tmp_path / "prepare.yaml"
        write_config_template(str(path), "prepare", overwrite=True)
        assert path.exists()

    def test_create_dirs_and_overwrite_together(self, tmp_path):
        """Both options apply to the same call."""
        path = tmp_path / "new" / "prepare.yaml"
        write_config_template(str(path), "prepare", create_dirs=True)
        write_config_template(str(path), "prepare", create_dirs=True, overwrite=True)
        assert path.exists()

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

    def test_tilde_path_is_expanded(self, tmp_path, monkeypatch):
        """A '~' path writes to the home directory, not a literal '~' folder."""
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.chdir(tmp_path)

        write_config_template("~/config/prepare.yaml", "prepare", create_dirs=True)

        assert (tmp_path / "config" / "prepare.yaml").exists()
        assert not (tmp_path / "~").exists()

    def test_missing_tilde_directory_reports_the_expanded_path(
        self, tmp_path, monkeypatch
    ):
        """The refusal names the real directory, so it can be created as shown."""
        monkeypatch.setenv("HOME", str(tmp_path))

        with pytest.raises(IOError) as excinfo:
            write_config_template("~/no_such_dir/prepare.yaml", "prepare")

        assert str(tmp_path / "no_such_dir") in str(excinfo.value)


#: (stage, extension, expected class) for every template the interface offers.
STAGE_SPECS = [
    ("prepare", "", DataSetConfig),
    ("prepare", "full", DataSetConfig),
    ("prepare", "reduced", DataSetConfig),
    ("train", "", TrainingConfig),
    ("classify", "", ClassificationConfig),
    ("classify", "full", ClassificationConfig),
    ("nrt_qc", "", NRTQCConfig),
]


class TestReadConfigTemplate:
    """Tests for ``read_config_template``."""

    @pytest.mark.parametrize(
        "stage, extension, expected_class",
        STAGE_SPECS,
        ids=[f"{stage}_{extension}" for stage, extension, _ in STAGE_SPECS],
    )
    def test_returns_a_resolved_config(self, stage, extension, expected_class):
        """Each stage returns its config class with the entry already selected."""
        config = read_config_template(stage, extension)

        assert isinstance(config, expected_class)
        assert config.data is not None
        assert config.dataset_name is not None

    @pytest.mark.parametrize(
        "stage, extension, _expected_class",
        STAGE_SPECS,
        ids=[f"{stage}_{extension}" for stage, extension, _ in STAGE_SPECS],
    )
    def test_matches_the_written_template(
        self, stage, extension, _expected_class, tmp_path
    ):
        """The object is built from the same YAML the file would contain.

        The two functions take the same arguments, so they must resolve the
        same template — they share one registry precisely so a stage cannot
        be added to one and forgotten in the other.

        Compared before selection: ``select()`` resolves the ``min_max``
        feature statistics into the configuration, so a selected object no
        longer matches the YAML it came from.
        """
        path = tmp_path / "template.yaml"
        write_config_template(str(path), stage, extension)

        written = yaml.safe_load(path.read_text())
        from_template = read_config_template(stage, extension, auto_select=False)
        assert from_template.full_config == written

    def test_without_auto_select(self):
        """Auto-selection can be switched off, leaving nothing resolved."""
        config = read_config_template("prepare", auto_select=False)

        assert config.data is None
        assert config.dataset_name is None
        assert config.full_config is not None

    def test_invalid_stage(self):
        """An unknown stage raises, as it does when writing."""
        with pytest.raises(ValueError):
            read_config_template("prepare2")

    def test_invalid_extension(self):
        """A known stage with an unknown variant raises too."""
        with pytest.raises(ValueError):
            read_config_template("train", "full")

    def test_returns_an_independent_object(self):
        """Each call builds a fresh object, so edits do not leak between them.

        The templates are module-level functions returning text; if they ever
        returned a shared structure instead, customizing one configuration
        would silently alter the next.
        """
        first = read_config_template("prepare")
        first.data["path_info"]["common"]["base_path"] = "/somewhere/else"

        second = read_config_template("prepare")
        assert second.data["path_info"]["common"]["base_path"] == "/path/to/data"


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
        """Naming a set in a multi-set file selects it.

        This used to raise: auto-selection ran first and rejected the file for
        holding several sets, so ``set_name`` was unusable without also passing
        ``auto_select=False``. Naming a set now switches auto-selection off,
        since there is nothing left for it to decide.
        """
        config = read_config(training_yaml_001, "NRT_BO_001")
        assert isinstance(config, TrainingConfig)
        assert config.dataset_name == "NRT_BO_001"

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

    def test_set_name_selects_from_a_multi_set_file(self, training_yaml_001):
        """Naming a set must work on a file holding several of them.

        Auto-selection rejects a file with more than one set, which is exactly
        the file a caller passing ``set_name`` is selecting from; naming a set
        has to switch it off rather than collide with it.
        """
        for name in ("NRT_BO_001", "NRT_BO_002"):
            config = read_config(training_yaml_001, set_name=name)
            assert config.dataset_name == name

    def test_auto_select_false_still_honoured(self, training_yaml_001):
        """The explicit three-argument form keeps working."""
        config = read_config(training_yaml_001, "NRT_BO_001", False)
        assert config.dataset_name == "NRT_BO_001"

    def test_multi_set_file_without_a_set_name_still_raises(self, training_yaml_001):
        """With nothing named there is still nothing to choose between."""
        with pytest.raises(ValueError, match="multiple data set names"):
            read_config(training_yaml_001)

    def test_config_with_invalid_path(self, dataset_yaml_001):
        """A non-existent file path raises IOError."""
        with pytest.raises(IOError):
            _ = read_config(str(dataset_yaml_001) + "abc")
