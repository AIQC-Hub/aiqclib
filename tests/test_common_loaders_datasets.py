"""Unit tests for the prepare-stage dataset loader functions.

Six classes, one per prepare-pipeline step:
- ``TestInputClassLoader``     — load_step1_input_dataset
- ``TestSummaryClassLoader``   — load_step2_summary_dataset
- ``TestSelectClassLoader``    — load_step3_select_dataset
- ``TestLocateClassLoader``    — load_step4_locate_dataset
- ``TestExtractClassLoader``   — load_step5_extract_dataset
- ``TestSplitClassLoader``     — load_step6_split_dataset

Each loader is exercised against two configs:
- ``dataset_config_001`` — standard prepare config (3-target, qc-filtered)
- ``dataset_config_005`` — "select-all" variant (keeps all rows, different
  loader classes for steps that have an _all suffix)

Refactored from already-pytest classes that used a ``_set_config(idx)`` helper
+ parallel data lists (e.g. ``selected_profiles = [50, 503]``) and indexed
into them by parametrize idx. The refactored version uses
``request.getfixturevalue()`` to resolve a fixture name to an instance, and
parametrize cases carry the per-config expected values directly — no more
indexing parallel lists.
"""

import polars as pl
import pytest

from aiqclib.common.loader.dataset_loader import (
    load_step1_input_dataset,
    load_step2_summary_dataset,
    load_step3_select_dataset,
    load_step4_locate_dataset,
    load_step5_extract_dataset,
    load_step6_split_dataset,
)
from aiqclib.prepare.step1_read_input.dataset_a import InputDataSetA
from aiqclib.prepare.step2_calc_stats.dataset_a import SummaryDataSetA
from aiqclib.prepare.step3_select_profiles.dataset_a import SelectDataSetA
from aiqclib.prepare.step3_select_profiles.dataset_all import SelectDataSetAll
from aiqclib.prepare.step4_select_rows.dataset_a import LocateDataSetA
from aiqclib.prepare.step4_select_rows.dataset_all import LocateDataSetAll
from aiqclib.prepare.step5_extract_features.dataset_a import ExtractDataSetA
from aiqclib.prepare.step6_split_dataset.dataset_a import SplitDataSetA
from aiqclib.prepare.step6_split_dataset.dataset_all import SplitDataSetAll


# ---------------------------------------------------------------------------
# Shared input-data expectations
# ---------------------------------------------------------------------------

# The full input parquet is the same for both configs (the configs differ
# in profile-selection strategy, not input data). All loaders that examine
# ``ds.input_data`` see this shape.
_INPUT_ROWS = 3267
_INPUT_COLS = 30


# ---------------------------------------------------------------------------
# Step 1: input
# ---------------------------------------------------------------------------

# (config_fixture_name, expected_class) — both configs use the same input class.
_INPUT_LOADER_CASES = [
    ("dataset_config_001", InputDataSetA),
    ("dataset_config_005", InputDataSetA),
]


class TestInputClassLoader:
    """Tests for load_step1_input_dataset."""

    @pytest.mark.parametrize(
        "config_fixture, expected_class",
        _INPUT_LOADER_CASES,
        ids=[case[0] for case in _INPUT_LOADER_CASES],
    )
    def test_load_dataset_valid_config(self, config_fixture, expected_class, request):
        """Default config produces the expected loader class with step_name='input'."""
        config = request.getfixturevalue(config_fixture)
        ds = load_step1_input_dataset(config)
        assert isinstance(ds, expected_class)
        assert ds.step_name == "input"

    def test_load_input_class_with_invalid_config(self, dataset_config_001):
        """An invalid input-class name raises ValueError."""
        dataset_config_001.data["step_class_set"]["steps"]["input"] = "InvalidClass"
        with pytest.raises(ValueError):
            _ = load_step1_input_dataset(dataset_config_001)


# ---------------------------------------------------------------------------
# Step 2: summary
# ---------------------------------------------------------------------------

_SUMMARY_LOADER_CASES = [
    ("dataset_config_001", SummaryDataSetA),
    ("dataset_config_005", SummaryDataSetA),
]


class TestSummaryClassLoader:
    """Tests for load_step2_summary_dataset."""

    @pytest.mark.parametrize(
        "config_fixture, expected_class",
        _SUMMARY_LOADER_CASES,
        ids=[case[0] for case in _SUMMARY_LOADER_CASES],
    )
    def test_load_dataset_valid_config(self, config_fixture, expected_class, request):
        """Default config produces the expected loader class with step_name='summary'."""
        config = request.getfixturevalue(config_fixture)
        ds = load_step2_summary_dataset(config)
        assert isinstance(ds, expected_class)
        assert ds.step_name == "summary"

    @pytest.mark.parametrize(
        "config_fixture, expected_class",
        _SUMMARY_LOADER_CASES,
        ids=[case[0] for case in _SUMMARY_LOADER_CASES],
    )
    def test_load_dataset_input_data(
        self, config_fixture, expected_class, request, test_data_file,
    ):
        """Provided input_data propagates to the loader with the expected shape."""
        config = request.getfixturevalue(config_fixture)

        ds_input = load_step1_input_dataset(config)
        ds_input.input_file_name = str(test_data_file)
        ds_input.read_input_data()

        ds = load_step2_summary_dataset(config, ds_input.input_data)
        assert isinstance(ds, expected_class)
        assert isinstance(ds.input_data, pl.DataFrame)
        assert ds.input_data.shape[0] == _INPUT_ROWS
        assert ds.input_data.shape[1] == _INPUT_COLS


# ---------------------------------------------------------------------------
# Step 3: select
# ---------------------------------------------------------------------------

# (config_fixture, expected_class). Note that step 3 has different loader
# classes per config (SelectDataSetA for the standard config, SelectDataSetAll
# for the select-all variant).
_SELECT_LOADER_CASES = [
    ("dataset_config_001", SelectDataSetA),
    ("dataset_config_005", SelectDataSetAll),
]


class TestSelectClassLoader:
    """Tests for load_step3_select_dataset."""

    @pytest.mark.parametrize(
        "config_fixture, expected_class",
        _SELECT_LOADER_CASES,
        ids=[case[0] for case in _SELECT_LOADER_CASES],
    )
    def test_load_dataset_valid_config(self, config_fixture, expected_class, request):
        """Each config produces its corresponding select loader class."""
        config = request.getfixturevalue(config_fixture)
        ds = load_step3_select_dataset(config)
        assert isinstance(ds, expected_class)
        assert ds.step_name == "select"

    @pytest.mark.parametrize(
        "config_fixture, expected_class",
        _SELECT_LOADER_CASES,
        ids=[case[0] for case in _SELECT_LOADER_CASES],
    )
    def test_load_dataset_input_data(
        self, config_fixture, expected_class, request, test_data_file,
    ):
        """Provided input_data propagates with the expected shape."""
        config = request.getfixturevalue(config_fixture)

        ds_input = load_step1_input_dataset(config)
        ds_input.input_file_name = str(test_data_file)
        ds_input.read_input_data()

        ds = load_step3_select_dataset(config, ds_input.input_data)
        assert isinstance(ds, expected_class)
        assert isinstance(ds.input_data, pl.DataFrame)
        assert ds.input_data.shape[0] == _INPUT_ROWS
        assert ds.input_data.shape[1] == _INPUT_COLS


# ---------------------------------------------------------------------------
# Step 4: locate
# ---------------------------------------------------------------------------

# (config_fixture, expected_class, expected_n_profiles)
# n_profiles differs: standard config selects 50 profiles (positive/negative
# pairs); select-all variant selects all 503 profiles.
# TODO: update after data reduction.
_LOCATE_LOADER_CASES = [
    ("dataset_config_001", LocateDataSetA,   14),
    ("dataset_config_005", LocateDataSetAll, 12),
]


class TestLocateClassLoader:
    """Tests for load_step4_locate_dataset."""

    @pytest.mark.parametrize(
        "config_fixture, expected_class, expected_n_profiles",
        _LOCATE_LOADER_CASES,
        ids=[case[0] for case in _LOCATE_LOADER_CASES],
    )
    def test_load_dataset_valid_config(
        self, config_fixture, expected_class, expected_n_profiles, request,
    ):
        """Each config produces its corresponding locate loader class."""
        config = request.getfixturevalue(config_fixture)
        ds = load_step4_locate_dataset(config)
        assert isinstance(ds, expected_class)
        assert ds.step_name == "locate"

    @pytest.mark.parametrize(
        "config_fixture, expected_class, expected_n_profiles",
        _LOCATE_LOADER_CASES,
        ids=[case[0] for case in _LOCATE_LOADER_CASES],
    )
    def test_load_dataset_input_data_and_profiles(
        self, config_fixture, expected_class, expected_n_profiles, request,
        test_data_file,
    ):
        """Provided input_data and selected_profiles propagate with expected shapes."""
        config = request.getfixturevalue(config_fixture)

        ds_input = load_step1_input_dataset(config)
        ds_input.input_file_name = str(test_data_file)
        ds_input.read_input_data()

        ds_select = load_step3_select_dataset(config, ds_input.input_data)
        ds_select.label_profiles()

        ds = load_step4_locate_dataset(
            config, ds_input.input_data, ds_select.selected_profiles,
        )
        assert isinstance(ds, expected_class)

        assert isinstance(ds.input_data, pl.DataFrame)
        assert ds.input_data.shape[0] == _INPUT_ROWS
        assert ds.input_data.shape[1] == _INPUT_COLS

        assert isinstance(ds.selected_profiles, pl.DataFrame)
        assert ds.selected_profiles.shape[0] == expected_n_profiles
        assert ds.selected_profiles.shape[1] == 8


# ---------------------------------------------------------------------------
# Step 5: extract
# ---------------------------------------------------------------------------

# (config_fixture, n_profiles, n_filtered, n_rows_temp, n_rows_psal).
# Both configs share ExtractDataSetA as the loader class (there's no
# ExtractDataSetAll). TODO: update values after data reduction.
_EXTRACT_LOADER_CASES = [
    ("dataset_config_001", 14,  2879,  24,    36),
    ("dataset_config_005", 12, 3267, 3267, 3267),
]


class TestExtractClassLoader:
    """Tests for load_step5_extract_dataset."""

    @pytest.mark.parametrize(
        "config_fixture, n_profiles, n_filtered, n_rows_temp, n_rows_psal",
        _EXTRACT_LOADER_CASES,
        ids=[case[0] for case in _EXTRACT_LOADER_CASES],
    )
    def test_load_dataset_valid_config(
        self, config_fixture, n_profiles, n_filtered, n_rows_temp, n_rows_psal,
        request,
    ):
        """Default config produces an ExtractDataSetA with step_name='extract'."""
        config = request.getfixturevalue(config_fixture)
        ds = load_step5_extract_dataset(config)
        assert isinstance(ds, ExtractDataSetA)
        assert ds.step_name == "extract"

    @pytest.mark.parametrize(
        "config_fixture, n_profiles, n_filtered, n_rows_temp, n_rows_psal",
        _EXTRACT_LOADER_CASES,
        ids=[case[0] for case in _EXTRACT_LOADER_CASES],
    )
    def test_load_dataset_input_data_and_profiles(
        self, config_fixture, n_profiles, n_filtered, n_rows_temp, n_rows_psal,
        request, test_data_file,
    ):
        """Provided upstream outputs propagate with expected shapes."""
        config = request.getfixturevalue(config_fixture)

        ds_input = load_step1_input_dataset(config)
        ds_input.input_file_name = str(test_data_file)
        ds_input.read_input_data()

        ds_select = load_step3_select_dataset(config, ds_input.input_data)
        ds_select.label_profiles()

        ds_summary = load_step2_summary_dataset(config, ds_input.input_data)
        ds_summary.calculate_stats()

        ds_locate = load_step4_locate_dataset(
            config, ds_input.input_data, ds_select.selected_profiles,
        )
        ds_locate.process_targets()

        ds = load_step5_extract_dataset(
            config,
            ds_input.input_data,
            ds_select.selected_profiles,
            ds_locate.selected_rows,
            ds_summary.summary_stats,
        )

        assert isinstance(ds, ExtractDataSetA)

        assert isinstance(ds.input_data, pl.DataFrame)
        assert ds.input_data.shape[0] == _INPUT_ROWS
        assert ds.input_data.shape[1] == _INPUT_COLS

        assert isinstance(ds.summary_stats, pl.DataFrame)
        assert ds.summary_stats.shape[0] == 65
        assert ds.summary_stats.shape[1] == 12

        assert isinstance(ds.selected_profiles, pl.DataFrame)
        assert ds.selected_profiles.shape[0] == n_profiles
        assert ds.selected_profiles.shape[1] == 8

        assert isinstance(ds.filtered_input, pl.DataFrame)
        assert ds.filtered_input.shape[0] == n_filtered
        assert ds.filtered_input.shape[1] == _INPUT_COLS

        assert isinstance(ds.selected_rows["temp"], pl.DataFrame)
        assert ds.selected_rows["temp"].shape[0] == n_rows_temp
        assert ds.selected_rows["temp"].shape[1] == 9

        assert isinstance(ds.selected_rows["psal"], pl.DataFrame)
        assert ds.selected_rows["psal"].shape[0] == n_rows_psal
        assert ds.selected_rows["psal"].shape[1] == 9


# ---------------------------------------------------------------------------
# Step 6: split
# ---------------------------------------------------------------------------

# (config_fixture, expected_class, n_target_features_temp, n_target_features_psal).
# TODO: update after data reduction.
_SPLIT_LOADER_CASES = [
    ("dataset_config_001", SplitDataSetA,   24,    36),
    ("dataset_config_005", SplitDataSetAll, 3267, 3267),
]


class TestSplitClassLoader:
    """Tests for load_step6_split_dataset."""

    @pytest.mark.parametrize(
        "config_fixture, expected_class, n_features_temp, n_features_psal",
        _SPLIT_LOADER_CASES,
        ids=[case[0] for case in _SPLIT_LOADER_CASES],
    )
    def test_load_dataset_valid_config(
        self, config_fixture, expected_class, n_features_temp, n_features_psal,
        request,
    ):
        """Each config produces its corresponding split loader class."""
        config = request.getfixturevalue(config_fixture)
        ds = load_step6_split_dataset(config)
        assert isinstance(ds, expected_class)
        assert ds.step_name == "split"

    @pytest.mark.parametrize(
        "config_fixture, expected_class, n_features_temp, n_features_psal",
        _SPLIT_LOADER_CASES,
        ids=[case[0] for case in _SPLIT_LOADER_CASES],
    )
    def test_load_dataset_input_data(
        self, config_fixture, expected_class, n_features_temp, n_features_psal,
        request, test_data_file,
    ):
        """Provided target_features propagate with expected per-target shapes."""
        config = request.getfixturevalue(config_fixture)

        ds_input = load_step1_input_dataset(config)
        ds_input.input_file_name = str(test_data_file)
        ds_input.read_input_data()

        ds_select = load_step3_select_dataset(config, ds_input.input_data)
        ds_select.label_profiles()

        ds_summary = load_step2_summary_dataset(config, ds_input.input_data)
        ds_summary.calculate_stats()

        ds_locate = load_step4_locate_dataset(
            config, ds_input.input_data, ds_select.selected_profiles,
        )
        ds_locate.process_targets()

        ds_extract = load_step5_extract_dataset(
            config,
            ds_input.input_data,
            ds_select.selected_profiles,
            ds_locate.selected_rows,
            ds_summary.summary_stats,
        )
        ds_extract.process_targets()

        ds = load_step6_split_dataset(config, ds_extract.target_features)

        assert isinstance(ds, expected_class)

        assert isinstance(ds.target_features["temp"], pl.DataFrame)
        assert ds.target_features["temp"].shape[0] == n_features_temp
        assert ds.target_features["temp"].shape[1] == 58

        assert isinstance(ds.target_features["psal"], pl.DataFrame)
        assert ds.target_features["psal"].shape[0] == n_features_psal
        assert ds.target_features["psal"].shape[1] == 58