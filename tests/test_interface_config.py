"""
Unit tests for configuration management functionalities,
including writing configuration templates and reading existing configuration files.
"""

import os
import pytest
from pathlib import Path

from aiqclib.common.config.classify_config import ClassificationConfig
from aiqclib.common.config.dataset_config import DataSetConfig
from aiqclib.common.config.training_config import TrainingConfig
from aiqclib.interface.config import read_config
from aiqclib.interface.config import write_config_template


class TestTemplateConfig:
    """
    Tests for verifying that configuration templates can be correctly
    written to disk for 'prepare' (dataset) and 'train' modules.
    """

    @pytest.fixture(autouse=True)
    def setup_clearup(self):
        """
        Set up test environment by defining sample file paths
        for dataset, training, and classification configuration templates.
        """
        config_path = Path(__file__).resolve().parent / "data" / "config"
        self.templates = [
            ("prepare", "", config_path / "temp_dataset_template.yaml"),
            ("prepare", "full", config_path / "temp_dataset_full_template.yaml"),
            (
                "prepare",
                "reduced",
                config_path / "temp_dataset_reduced_template.yaml",
            ),
            ("train", "", config_path / "temp_training_template.yaml"),
            ("classify", "", config_path / "temp_classification_template.yaml"),
            (
                "classify",
                "full",
                config_path / "temp_classification_template.yaml",
            ),
        ]

    @pytest.mark.parametrize("idx", range(6))
    def test_write_config_template(self, idx):
        """
        Check that a configuration template can be written
        to the specified path and removed afterward.
        """
        write_config_template(
            self.templates[idx][2], self.templates[idx][0], self.templates[idx][1]
        )
        assert os.path.exists(self.templates[idx][2])
        os.remove(self.templates[idx][2])

    def test_config_template_with_invalid_module(self):
        """
        Ensure that requesting a template for an invalid module name
        raises ValueError.
        """
        with pytest.raises(ValueError):
            write_config_template(self.templates[0][2], "prepare2")

    def test_config_template_with_invalid_path(self):
        """
        Ensure that attempting to write a template to an invalid path
        raises IOError.
        """
        with pytest.raises(IOError):
            write_config_template("/abc" + str(self.templates[0][2]), "prepare")


class TestReadConfig:
    """
    Tests for verifying that reading an existing config file returns
    the appropriate DataSetConfig or TrainingConfig object, while
    invalid inputs raise errors.
    """

    @pytest.fixture(autouse=True)
    def setup_clearup(self):
        """
        Define sample file paths for dataset, training, and classification
        configuration files used in subsequent tests.
        """

        dir_config = Path(__file__).resolve().parent / "data" / "config"
        self.ds_config_files = [
            dir_config / "test_dataset_001.yaml",
            dir_config / "test_dataset_004.yaml",
        ]

        self.train_config_file = dir_config / "test_training_001.yaml"

        self.classification_config_files = [
            dir_config / "test_classify_001.yaml",
            dir_config / "test_classify_002.yaml",
        ]

        self.invalid_config_file = dir_config / "test_dataset_invalid.yaml"

    @pytest.mark.parametrize("idx", range(2))
    def test_ds_config(self, idx):
        """
        Verify that reading a dataset (prepare) config file returns
        a DataSetConfig instance.
        """
        config = read_config(self.ds_config_files[idx])
        assert isinstance(config, DataSetConfig)

    def test_train_config(self):
        """
        Verify that reading a training config file returns
        a TrainingConfig instance.
        """
        config = read_config(self.train_config_file)
        assert isinstance(config, TrainingConfig)

    @pytest.mark.parametrize("idx", range(2))
    def test_classify_config(self, idx):
        """
        Verify that reading a classification config file returns
        a ClassificationConfig instance.
        """
        config = read_config(self.classification_config_files[idx])
        assert isinstance(config, ClassificationConfig)

    def test_config_with_invalid_module(self):
        """
        Check that specifying an invalid module name (config_type within file)
        raises ValueError.
        """
        with pytest.raises(ValueError):
            _ = read_config(self.invalid_config_file)

    def test_config_with_invalid_path(self):
        """
        Check that providing an invalid file path raises IOError.
        """
        with pytest.raises(IOError):
            _ = read_config(str(self.ds_config_files[0]) + "abc")
