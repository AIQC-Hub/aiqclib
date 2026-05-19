"""Unit tests for the ``load_feature_class`` loader.

load_feature_class is the factory that turns a feature-param dict (one
entry from ``config.data["feature_param_set"]["params"]``) into a concrete
feature object. The first entry in ``test_dataset_001.yaml``'s
feature_param_set defines a ``LocationFeat``-class feature — the tests
verify the loader returns the right instance for that entry and raises
ValueError when given an invalid feature name.

Refactored from a ``unittest.TestCase`` class with the standard setUp
boilerplate. Uses ``dataset_config_001`` from conftest.
"""

import pytest

from aiqclib.common.loader.feature_loader import load_feature_class
from aiqclib.prepare.features.location import LocationFeat


class TestFeatureClassLoader:
    """Tests for the load_feature_class factory."""

    def test_load_model_valid_config(self, dataset_config_001):
        """A valid feature-param entry produces an instance of the configured class."""
        ds = load_feature_class(
            "temp", dataset_config_001.data["feature_param_set"]["params"][0],
        )
        assert isinstance(ds, LocationFeat)

    def test_load_model_invalid_config(self, dataset_config_001):
        """An invalid feature name in the param entry raises ValueError."""
        dataset_config_001.data["feature_param_set"]["params"][0]["feature"] = (
            "invalid_feature_name"
        )
        with pytest.raises(ValueError):
            _ = load_feature_class(
                "temp", dataset_config_001.data["feature_param_set"]["params"][0],
            )