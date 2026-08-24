"""
Tests for the feature-level metadata used by the profile-level pipeline.
"""

from aiqclib.common.base.feature_base import FeatureBase
from aiqclib.common.loader.feature_registry import FEATURE_REGISTRY

# The features that produce one value per profile; everything else in the
# registry is observation-level. qc_pressure_increasing is deliberately
# observation-level despite its profile-style column name (its flag varies
# within a profile).
PROFILE_LEVEL_FEATURES = {
    "location",
    "day_of_year",
    "profile_summary_stats",
    "qc_impossible_date",
    "qc_impossible_location",
    "qc_position_on_land",
    "qc_stuck_value",
}


class TestFeatureLevels:
    def test_default_level_is_observation(self):
        assert FeatureBase.level == "observation"

    def test_every_registry_entry_has_valid_level(self):
        for name, cls in FEATURE_REGISTRY.items():
            assert cls.level in {"observation", "profile"}, name

    def test_profile_level_features(self):
        actual = {
            name for name, cls in FEATURE_REGISTRY.items() if cls.level == "profile"
        }
        assert actual == PROFILE_LEVEL_FEATURES
