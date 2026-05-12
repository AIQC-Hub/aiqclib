"""Unit tests for the ``read_config`` function.

Verifies that ``read_config`` can load YAML files via explicit paths, raises
``TypeError`` when called with no arguments, and raises ``FileNotFoundError``
when the path doesn't exist.

This is a "pure logic" test file: it exercises ``read_config`` itself rather
than the data pipeline. The only fixture it touches is a path to an existing
test YAML — the content of that YAML doesn't matter beyond having the
expected top-level keys.
"""

import pytest

from aiqclib.common.utils.config import read_config


class TestReadConfig:
    """Tests for ``read_config``: explicit path, missing args, missing file."""

    def test_read_config_with_explicit_file(self, dataset_yaml_001):
        """Loading by explicit path returns a dict with expected top-level keys."""
        config = read_config(config_file=str(dataset_yaml_001))
        assert config is not None
        assert "data_sets" in config
        assert "path_info_sets" in config

    def test_read_config_no_params_raises_error(self):
        """Calling with neither ``config_file`` nor ``config_file_name`` raises TypeError."""
        with pytest.raises(TypeError):
            read_config()

    def test_read_config_nonexistent_file(self):
        """A non-existent path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            read_config(config_file="non_existent.yaml")
