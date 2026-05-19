"""Unit tests for the path/target/summary-stats methods on ``DataSetConfig``.

These tests exercise the ConfigBase-derived helpers that resolve paths,
look up step base classes, retrieve target-variable metadata, and read
summary-stats blocks from the loaded YAML. The DataSetConfig class is
used as a concrete instance, but the methods under test live on
ConfigBase (and are inherited identically by TrainingConfig and
ClassificationConfig).

Four test classes, organised by concern:
- ``TestBaseConfigPathMethods``    — get_base_path / get_step_folder_name /
  get_dataset_folder_name / get_file_name / get_full_file_name
- ``TestBaseConfigBaseClass``      — get_base_class
- ``TestBaseConfigTargets``        — get_target_variables / get_target_names
  / get_target_dict / get_target_file_names
- ``TestBaseConfigSummaryStats``   — get_summary_stats / feature param
  update

Refactored from four ``unittest.TestCase`` classes that each repeated the
same setUp boilerplate. All four classes now use ``dataset_config_001`` or
``dataset_config_002`` from conftest; setUp methods are gone.
"""

import pytest


# ---------------------------------------------------------------------------
# Path-related methods
# ---------------------------------------------------------------------------


class TestBaseConfigPathMethods:
    """Tests for path-resolution methods on DataSetConfig.

    Most tests use ``dataset_config_001``; a few that need overridden step
    parameters (e.g. ``test_default_base_path``) use ``dataset_config_002``.
    """

    def test_common_base_path(self, dataset_config_001):
        """get_base_path('common') returns the configured common base path."""
        assert dataset_config_001.get_base_path("common") == "/path/to/data_1"

    def test_input_base_path(self, dataset_config_001):
        """get_base_path('input') returns the step-specific base path when present."""
        assert dataset_config_001.get_base_path("input") == "/path/to/input_1"

    def test_default_base_path(self, dataset_config_002):
        """get_base_path falls back to the common path when no step-specific one exists.

        Uses config 002 because that's where 'locate' has no dedicated base_path.
        """
        assert dataset_config_002.get_base_path("locate") == "/path/to/data_1"

    def test_input_step_folder_name(self, dataset_config_001):
        """get_step_folder_name('input') returns the configured folder name."""
        assert dataset_config_001.get_step_folder_name("input") == "input_folder_1"

    def test_auto_select_step_folder_name(self, dataset_config_002):
        """When no folder name is configured, auto-select returns the step name itself."""
        assert dataset_config_002.get_step_folder_name("select") == "select"

    def test_no_auto_select_step_folder_name(self, dataset_config_002):
        """With folder_name_auto=False, the step name is *not* used as a fallback —
        the result is an empty string when no folder name is configured."""
        assert (
            dataset_config_002.get_step_folder_name("select", folder_name_auto=False)
            == ""
        )

    def test_common_dataset_folder_name(self, dataset_config_001):
        """get_dataset_folder_name('input') returns the top-level dataset folder."""
        assert dataset_config_001.get_dataset_folder_name("input") == "nrt_bo_001"

    def test_dataset_folder_name_in_step_params(self, dataset_config_002):
        """A step can override the dataset folder name via its step_param entry."""
        assert (
            dataset_config_002.get_dataset_folder_name("summary")
            == "summary_dataset_folder"
        )

    def test_default_file_name(self, dataset_config_001):
        """get_file_name returns the provided default when no file name is configured."""
        assert (
            dataset_config_001.get_file_name("input", "default_file.txt")
            == "default_file.txt"
        )

    def test_no_default_file_name(self, dataset_config_001):
        """get_file_name with no default and no configured name raises ValueError."""
        with pytest.raises(ValueError):
            _ = dataset_config_001.get_file_name("input")

    def test_file_name_in_params(self, dataset_config_002):
        """A step can specify its own file name via step_params."""
        assert dataset_config_002.get_file_name("summary") == "summary_in_params.txt"

    def test_full_input_path(self, dataset_config_001):
        """get_full_file_name with use_dataset_folder=False omits the dataset folder."""
        assert (
            dataset_config_001.get_full_file_name(
                "input",
                "test_input_file.txt",
                use_dataset_folder=False,
            )
            == "/path/to/input_1/input_folder_1/test_input_file.txt"
        )

    def test_full_input_path_with_dataset_folder(self, dataset_config_001):
        """get_full_file_name with default args includes the dataset folder."""
        assert (
            dataset_config_001.get_full_file_name("input", "test_input_file.txt")
            == "/path/to/input_1/nrt_bo_001/input_folder_1/test_input_file.txt"
        )

    def test_full_summary_path(self, dataset_config_002):
        """End-to-end path resolution with step-param overrides for folder and file name."""
        assert (
            dataset_config_002.get_full_file_name("summary", "test_input_file.txt")
            == "/path/to/data_1/summary_dataset_folder/summary/summary_in_params.txt"
        )


# ---------------------------------------------------------------------------
# Base-class lookup
# ---------------------------------------------------------------------------


class TestBaseConfigBaseClass:
    """Tests for get_base_class — the step → class-name mapping from step_class_set."""

    def test_input_base_class(self, dataset_config_001):
        """get_base_class('input') returns the configured class name (InputDataSetA)."""
        assert dataset_config_001.get_base_class("input") == "InputDataSetA"


# ---------------------------------------------------------------------------
# Target-variable lookups
# ---------------------------------------------------------------------------

# Expected per-target dict for the 3-target test_dataset_001.yaml config.
# Module-level constant: defines the contract being tested. If the YAML
# schema changes, update here.
_EXPECTED_TARGET_DICTS = {
    "temp": {
        "name": "temp",
        "flag": "temp_qc",
        "pos_flag_values": [4],
        "neg_flag_values": [1],
    },
    "psal": {
        "name": "psal",
        "flag": "psal_qc",
        "pos_flag_values": [4],
        "neg_flag_values": [1],
    },
    "pres": {
        "name": "pres",
        "flag": "pres_qc",
        "pos_flag_values": [4],
        "neg_flag_values": [1],
    },
}


class TestBaseConfigTargets:
    """Tests for target-variable accessors.

    These tests inspect the 3-target ``target_set_1`` defined in
    test_dataset_001.yaml (temp/psal/pres). The dataset config still
    includes pres — only the classify-side YAMLs were updated to use a
    2-target ``target_set_1_2``. The dataset target_set is independent
    of whether pres data is empty in the train/test split, so all three
    targets remain in the config.
    """

    def test_target_variables(self, dataset_config_001):
        """get_target_variables returns one entry per configured target."""
        target_variables = dataset_config_001.get_target_variables()
        assert len(target_variables) == 3

    def test_target_names(self, dataset_config_001):
        """get_target_names returns the configured target names in order."""
        assert dataset_config_001.get_target_names() == ["temp", "psal", "pres"]

    def test_target_dict(self, dataset_config_001):
        """get_target_dict returns the full per-target metadata dict."""
        target_dict = dataset_config_001.get_target_dict()
        for tgt, expected in _EXPECTED_TARGET_DICTS.items():
            assert target_dict[tgt] == expected

    def test_target_file_names(self, dataset_config_001):
        """get_target_file_names expands the ``{target_name}`` placeholder per target."""
        target_file_names = dataset_config_001.get_target_file_names(
            "select",
            "{target_name}_features.parquet",
        )
        base = "/path/to/select_1/nrt_bo_001/select_folder_1"
        for tgt in _EXPECTED_TARGET_DICTS:
            assert target_file_names[tgt] == f"{base}/{tgt}_features.parquet"


# ---------------------------------------------------------------------------
# Summary-stats lookups
# ---------------------------------------------------------------------------


class TestBaseConfigSummaryStats:
    """Tests for get_summary_stats and the feature-param update side-effect."""

    def test_config_location_summary_stats(self, dataset_config_001):
        """get_summary_stats('location') returns longitude/latitude entries."""
        stats = dataset_config_001.get_summary_stats("location")
        assert "longitude" in stats
        assert "latitude" in stats

    def test_config_profile_summary_stats(self, dataset_config_001):
        """get_summary_stats('profile_summary_stats') returns per-variable stat blocks."""
        stats = dataset_config_001.get_summary_stats("profile_summary_stats")
        assert "pres" in stats
        assert "mean" in stats["pres"]
        assert "median" in stats["pres"]

    def test_config_basic_values3_stats(self, dataset_config_001):
        """get_summary_stats('basic_values3') returns min/max entries per variable."""
        stats = dataset_config_001.get_summary_stats("basic_values3")
        assert "pres" in stats
        assert "min" in stats["pres"]
        assert stats["pres"]["min"] == 0

    def test_update_feature_param_with_stats(self, dataset_config_001):
        """Every feature_param entry that has a stats_set has been augmented with stats.

        This side-effect happens during ``.select()`` — feature_param_set
        entries with a stats_set reference get a populated ``stats`` field
        from the corresponding summary_stats block.
        """
        for x in dataset_config_001.data["feature_param_set"]["params"]:
            if "stats_set" in x:
                assert "stats" in x
