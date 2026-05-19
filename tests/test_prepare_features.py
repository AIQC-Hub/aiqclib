"""Unit tests for the feature-extraction classes in
``aiqclib.prepare.features``.

Each feature class (LocationFeat, DayOfYearFeat, ProfileSummaryStats,
BasicValues) consumes the same five upstream inputs (selected_profiles,
filtered_input, selected_rows, summary_stats) and produces feature
DataFrames. Tests verify:
- Input wiring (the upstream frames land on the feature instance with
  expected shapes) — shared across all features via
  ``_assert_init_arguments``.
- Feature-specific extraction logic (extract_features + scale_first /
  scale_second produce the expected output shape).

Refactored from a base class ``_TestFeatureBase`` with ``_setup`` and
``_test_init_arguments`` helper methods, plus four subclasses. The
inheritance hierarchy doesn't translate cleanly to pytest, so:
- The shared ``pipeline`` fixture is defined at module scope.
- The shared ``_assert_init_arguments`` is a module-level helper.
- Each feature still has its own test class for its specific tests, but
  the class-level inheritance is gone.
"""

import polars as pl
import pytest

from aiqclib.prepare.features.basic_values import BasicValues
from aiqclib.prepare.features.day_of_year import DayOfYearFeat
from aiqclib.prepare.features.location import LocationFeat
from aiqclib.prepare.features.profile_summary import ProfileSummaryStats

from tests.conftest import build_prepare_pipeline


# ---------------------------------------------------------------------------
# Shared fixtures and helper
# ---------------------------------------------------------------------------


@pytest.fixture
def pipeline(dataset_config_001, test_data_file):
    """Prepare-pipeline output through step5 (extract) for config 001.

    Used by every feature-test class. The feature classes take their inputs
    from this pipeline's stages: ``selected_profiles`` from step3,
    ``filtered_input`` from step5, ``selected_rows`` from step4,
    ``summary_stats`` from step2.

    Note: pipeline.extract is built but ``process_targets()`` is *not*
    called on it. The original ``_setup`` helper also omitted that call,
    because the feature classes use ``filtered_input`` (a step5
    pre-processing output) but don't need the final ``target_features``.
    """
    return build_prepare_pipeline(
        dataset_config_001,
        test_data_file,
        stop_after="extract",
    )


def _make_feature(feature_cls, feature_info, pipeline, target: str = "temp"):
    """Construct a feature class instance from a build_prepare_pipeline result.

    Pure helper — no test state. Used by every per-feature test that
    instantiates a feature class.
    """
    return feature_cls(
        target,
        feature_info,
        pipeline.select.selected_profiles,
        pipeline.extract.filtered_input,
        pipeline.locate.selected_rows,
        pipeline.summary.summary_stats,
    )


def _assert_init_arguments(ds) -> None:
    """Assert the shape of every upstream input on a feature instance.

    Used by each feature class's ``test_init_arguments`` test. Replaces
    the original ``_TestFeatureBase._test_init_arguments``.
    """
    assert isinstance(ds.selected_profiles, pl.DataFrame)
    assert ds.selected_profiles.shape[0] == 14
    assert ds.selected_profiles.shape[1] == 8

    assert isinstance(ds.filtered_input, pl.DataFrame)
    assert ds.filtered_input.shape[0] == 2879
    assert ds.filtered_input.shape[1] == 30

    assert isinstance(ds.selected_rows["temp"], pl.DataFrame)
    assert ds.selected_rows["temp"].shape[0] == 24
    assert ds.selected_rows["temp"].shape[1] == 9

    assert isinstance(ds.selected_rows["psal"], pl.DataFrame)
    assert ds.selected_rows["psal"].shape[0] == 36
    assert ds.selected_rows["psal"].shape[1] == 9

    assert isinstance(ds.summary_stats, pl.DataFrame)
    assert ds.summary_stats.shape[0] == 65
    assert ds.summary_stats.shape[1] == 12


# ---------------------------------------------------------------------------
# LocationFeat
# ---------------------------------------------------------------------------

LOCATION_FEATURE_INFO = {
    "class": "location",
    "stats": {
        "longitude": {"min": 14.5, "max": 23.5},
        "latitude": {"min": 55, "max": 66},
    },
    "col_names": ["longitude", "latitude"],
    "stats_set": {"type": "min_max", "name": "location"},
}


class TestLocationFeature:
    """Tests for the LocationFeat class (longitude / latitude scaling)."""

    def test_init_arguments(self, pipeline):
        """Upstream input shapes pass through to the feature instance."""
        ds = _make_feature(LocationFeat, LOCATION_FEATURE_INFO, pipeline)
        _assert_init_arguments(ds)

    def test_location_features(self, pipeline):
        """extract_features + scale_second produces a 128×3 frame."""
        ds = _make_feature(LocationFeat, LOCATION_FEATURE_INFO, pipeline)
        ds.extract_features()
        ds.scale_second()

        assert isinstance(ds.features, pl.DataFrame)
        assert ds.features.shape[0] == 24
        assert ds.features.shape[1] == 3


# ---------------------------------------------------------------------------
# DayOfYearFeat
# ---------------------------------------------------------------------------

DAY_OF_YEAR_FEATURE_INFO = {
    "class": "day_of_year",
    "convert": "sine",
}


class TestDayOfYearFeature:
    """Tests for the DayOfYearFeat class (date-based features with sine/cosine)."""

    def test_init_arguments(self, pipeline):
        """Upstream input shapes pass through to the feature instance."""
        ds = _make_feature(DayOfYearFeat, DAY_OF_YEAR_FEATURE_INFO, pipeline)
        _assert_init_arguments(ds)

    def test_day_of_year_features(self, pipeline):
        """Sine conversion produces a 128×2 frame."""
        ds = _make_feature(DayOfYearFeat, DAY_OF_YEAR_FEATURE_INFO, pipeline)
        ds.extract_features()
        ds.scale_second()

        assert isinstance(ds.features, pl.DataFrame)
        assert ds.features.shape[0] == 24
        assert ds.features.shape[1] == 2

    def test_day_of_year_features_no_param(self, pipeline):
        """scale_second with feature_info=None is a no-op."""
        ds = _make_feature(DayOfYearFeat, DAY_OF_YEAR_FEATURE_INFO, pipeline)
        ds.extract_features()
        features = ds.features
        ds.feature_info = None  # Simulate missing feature info
        ds.scale_second()
        assert ds.features.equals(features)

    def test_day_of_year_features_no_convert_param(self, pipeline):
        """scale_second with no 'convert' key in feature_info is a no-op."""
        ds = _make_feature(DayOfYearFeat, DAY_OF_YEAR_FEATURE_INFO, pipeline)
        ds.extract_features()
        features = ds.features
        ds.feature_info = {"class": "day_of_year"}  # Simulate missing 'convert'
        ds.scale_second()
        assert ds.features.equals(features)

    def test_convert_cosine(self, pipeline):
        """Cosine conversion produces a 128×2 frame (same shape as sine)."""
        ds = _make_feature(DayOfYearFeat, DAY_OF_YEAR_FEATURE_INFO, pipeline)
        ds.feature_info = {
            "class": "day_of_year",
            "convert": "cosine",
        }
        ds.extract_features()
        ds.scale_second()

        assert isinstance(ds.features, pl.DataFrame)
        assert ds.features.shape[0] == 24
        assert ds.features.shape[1] == 2


# ---------------------------------------------------------------------------
# ProfileSummaryStats
# ---------------------------------------------------------------------------

PROFILE_SUMMARY_STATS_FEATURE_INFO = {
    "class": "profile_summary_stats",
    "stats": {
        "temp": {
            "mean": {"min": 0, "max": 12.5},
            "median": {"min": 0, "max": 15},
            "sd": {"min": 0, "max": 6.5},
            "pct25": {"min": 0, "max": 12},
            "pct75": {"min": 1, "max": 19},
        },
        "psal": {
            "mean": {"min": 2.9, "max": 12},
            "median": {"min": 2.9, "max": 12},
            "sd": {"min": 0, "max": 4},
            "pct25": {"min": 2.5, "max": 8.5},
            "pct75": {"min": 3, "max": 16},
        },
        "pres": {
            "mean": {"min": 24, "max": 105},
            "median": {"min": 24, "max": 105},
            "sd": {"min": 13, "max": 60},
            "pct25": {"min": 12, "max": 53},
            "pct75": {"min": 35, "max": 156},
        },
    },
    "col_names": ["temp", "psal", "pres"],
    "stats_set": {"type": "min_max", "name": "profile_summary_stats"},
    "summary_stats_names": ["mean", "median", "sd", "pct25", "pct75"],
}


class TestProfileSummaryStatsFeature:
    """Tests for the ProfileSummaryStats class (per-variable aggregate stats)."""

    def test_init_arguments(self, pipeline):
        """Upstream input shapes pass through to the feature instance."""
        ds = _make_feature(
            ProfileSummaryStats,
            PROFILE_SUMMARY_STATS_FEATURE_INFO,
            pipeline,
        )
        _assert_init_arguments(ds)

    def test_profile_summary_stats_features(self, pipeline):
        """extract_features + scale_second produces a 128×16 frame.

        16 columns = 3 variables × 5 stats + 1 (probably the target column).
        """
        ds = _make_feature(
            ProfileSummaryStats,
            PROFILE_SUMMARY_STATS_FEATURE_INFO,
            pipeline,
        )
        ds.extract_features()
        ds.scale_second()

        assert isinstance(ds.features, pl.DataFrame)
        assert ds.features.shape[0] == 24
        assert ds.features.shape[1] == 16


# ---------------------------------------------------------------------------
# BasicValues (basic_values3_plus_flanks variant)
# ---------------------------------------------------------------------------

BASIC_VALUES_FEATURE_INFO = {
    "class": "basic_values3_plus_flanks",
    "flank_up": 5,
    "stats": {
        "temp": {"min": 0, "max": 20},
        "psal": {"min": 0, "max": 20},
        "pres": {"min": 0, "max": 200},
    },
    "col_names": ["temp", "psal", "pres"],
    "stats_set": {"type": "min_max", "name": "basic_values3_plus_flanks"},
}


class TestBasicValues3PlusFlanksFeature:
    """Tests for the BasicValues class (basic stats + adjacent-point flanks)."""

    def test_init_arguments(self, pipeline):
        """Upstream input shapes pass through to the feature instance."""
        ds = _make_feature(BasicValues, BASIC_VALUES_FEATURE_INFO, pipeline)
        _assert_init_arguments(ds)

    def test_basic_values3_features(self, pipeline):
        """scale_first + extract_features produces a 128×4 frame."""
        ds = _make_feature(BasicValues, BASIC_VALUES_FEATURE_INFO, pipeline)

        # Note: this feature uses scale_first (not scale_second).
        ds.scale_first()
        ds.extract_features()

        assert isinstance(ds.features, pl.DataFrame)
        assert ds.features.shape[0] == 24
        assert ds.features.shape[1] == 4
