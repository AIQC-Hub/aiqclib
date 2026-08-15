"""Unit tests for the NRT QC module's step 2 (run QC items).

Runs the full template item set against the CTD test fixture and checks
the produced flag columns: presence, alignment, never-null values within
the flag scheme, and the deferral of the propagation item to step 3.
"""

import os

import polars as pl
import pytest

from aiqclib.common.loader.nrtqc_loader import (
    load_nrtqc_step1_input_dataset,
    load_nrtqc_step2_qc_dataset,
)

#: Flag columns expected from the template items (temp_to_psal deferred).
EXPECTED_ITEM_COLUMNS = {
    "qc_impossible_date",
    "qc_impossible_location",
    "qc_pressure_increasing",
    "temp_qc_global_range",
    "psal_qc_global_range",
    "temp_qc_regional_range",
    "psal_qc_regional_range",
    "temp_qc_spike",
    "psal_qc_spike",
    "temp_qc_gradient",
    "psal_qc_gradient",
    "temp_qc_digit_rollover",
    "psal_qc_digit_rollover",
    "temp_qc_stuck_value",
    "psal_qc_stuck_value",
    "temp_qc_density_inversion",
    "psal_qc_density_inversion",
}


@pytest.fixture
def qc_step(nrtqc_config_001):
    """Step 2 dataset with items already run on the test fixture."""
    ds_input = load_nrtqc_step1_input_dataset(nrtqc_config_001)
    ds_input.read_input_data()
    ds_qc = load_nrtqc_step2_qc_dataset(nrtqc_config_001, ds_input.input_data)
    ds_qc.run_qc_items()
    return ds_qc


class TestRunQCItems:
    """run_qc_items applies every enabled item to the fixture data."""

    def test_item_columns_added(self, qc_step):
        assert set(qc_step.qc_item_columns()) == EXPECTED_ITEM_COLUMNS

    def test_rows_and_input_columns_preserved(self, qc_step):
        assert qc_step.qc_data.shape[0] == qc_step.input_data.shape[0]
        assert set(qc_step.input_data.columns) <= set(qc_step.qc_data.columns)

    def test_flags_never_null_and_in_scheme(self, qc_step):
        for column in EXPECTED_ITEM_COLUMNS:
            flags = qc_step.qc_data[column]
            assert flags.null_count() == 0, column
            assert set(flags.unique().to_list()) <= {1, 3, 4}, column

    def test_regional_range_flags_baltic_temperatures(self, qc_step):
        """The Mediterranean template ranges must flag cold Baltic water."""
        flagged = qc_step.qc_data["temp_qc_regional_range"].to_list().count(4)
        assert flagged > 0

    def test_propagation_item_deferred(self, qc_step):
        """temp_to_psal is not run in step 2 (needs aggregated flags)."""
        assert "psal_qc_temp_to_psal" not in qc_step.qc_data.columns

    def test_items_excluded_from_the_final_flag_still_run(self, nrtqc_config_001):
        """include_in_final_flag governs aggregation only, never execution.

        Step 2 must keep producing every item's column so an excluded item
        stays usable as a training feature. Pinned because filtering here
        would look like a reasonable optimisation and would silently drop
        columns from the output.
        """
        by_name = {x["name"]: x for x in nrtqc_config_001.data["qc_item_set"]["items"]}
        for name in ("spike", "impossible_date", "global_range"):
            by_name[name]["include_in_final_flag"] = False

        ds_input = load_nrtqc_step1_input_dataset(nrtqc_config_001)
        ds_input.read_input_data()
        ds_qc = load_nrtqc_step2_qc_dataset(nrtqc_config_001, ds_input.input_data)
        ds_qc.run_qc_items()

        assert set(ds_qc.qc_item_columns()) == EXPECTED_ITEM_COLUMNS


class TestPositionOnLandThroughTheModule:
    """RTQC4 wired through step 2, which is where its column requirement bites.

    The item is not in the template, so it has to be added by name. The
    fixture carries no ``bathymetry`` column, which makes both paths
    testable: the error when it is absent, and the flags when one is
    supplied. It does carry ``bath``, real negative bathymetry, which the
    custom-column test uses as-is.
    """

    @staticmethod
    def _enable(config, params=None):
        """Append position_on_land to the configured item set."""
        entry = {"name": "position_on_land"}
        if params is not None:
            entry["params"] = params
        config.data["qc_item_set"]["items"].append(entry)

    @staticmethod
    def _run(config, input_data):
        ds_qc = load_nrtqc_step2_qc_dataset(config, input_data)
        ds_qc.run_qc_items()
        return ds_qc

    def test_missing_column_raises_through_the_step(self, nrtqc_config_001):
        """The fixture has no 'bathymetry' column, so enabling the item errors.

        Note the fixture *does* have 'bath'. Defaulting to a distinct name
        is what makes this an error rather than a silent misread.
        """
        self._enable(nrtqc_config_001)
        ds_input = load_nrtqc_step1_input_dataset(nrtqc_config_001)
        ds_input.read_input_data()

        with pytest.raises(
            ValueError, match="needs the sea floor depth column 'bathymetry'"
        ):
            self._run(nrtqc_config_001, ds_input.input_data)

    def test_flags_positions_on_land(self, nrtqc_config_001):
        """With the column present the item adds its profile-level flag."""
        self._enable(nrtqc_config_001)
        ds_input = load_nrtqc_step1_input_dataset(nrtqc_config_001)
        ds_input.read_input_data()

        # Half the rows put on land, so both outcomes appear.
        n = ds_input.input_data.shape[0]
        depths = [120.0 if i % 2 == 0 else -5.0 for i in range(n)]
        with_depth = ds_input.input_data.with_columns(
            pl.Series("bathymetry", depths, dtype=pl.Float64)
        )

        ds_qc = self._run(nrtqc_config_001, with_depth)
        flags = ds_qc.qc_data["qc_position_on_land"]
        assert "qc_position_on_land" in ds_qc.qc_item_columns()
        assert flags.null_count() == 0
        assert set(flags.unique().to_list()) == {1, 4}
        assert flags.to_list().count(4) == len([d for d in depths if d <= 0])

    def test_real_bath_column_with_negative_convention(self, nrtqc_config_001):
        """Both params reach the item, checked against the fixture's own data.

        'bath' is real externally computed bathymetry, negative below sea
        level, so with positive_depth False every Baltic row is in water.
        """
        self._enable(
            nrtqc_config_001,
            {"depth_column": "bath", "positive_depth": False},
        )
        ds_input = load_nrtqc_step1_input_dataset(nrtqc_config_001)
        ds_input.read_input_data()

        bath = ds_input.input_data["bath"]
        assert bath.max() < 0, "fixture precondition: bath is negative"

        ds_qc = self._run(nrtqc_config_001, ds_input.input_data)
        assert set(ds_qc.qc_data["qc_position_on_land"].unique().to_list()) == {1}

    def test_real_bath_column_read_with_the_wrong_convention(self, nrtqc_config_001):
        """Reading the same column as positive flags every row as land.

        The failure mode the two parameters exist to prevent, shown on real
        data rather than asserted in prose.
        """
        self._enable(nrtqc_config_001, {"depth_column": "bath"})
        ds_input = load_nrtqc_step1_input_dataset(nrtqc_config_001)
        ds_input.read_input_data()

        ds_qc = self._run(nrtqc_config_001, ds_input.input_data)
        assert set(ds_qc.qc_data["qc_position_on_land"].unique().to_list()) == {4}


class TestQCStepErrors:
    """Error handling of the QC step."""

    def test_unknown_item_raises(self, nrtqc_config_001):
        nrtqc_config_001.data["qc_item_set"]["items"].append({"name": "bogus"})
        ds_input = load_nrtqc_step1_input_dataset(nrtqc_config_001)
        ds_input.read_input_data()
        ds_qc = load_nrtqc_step2_qc_dataset(nrtqc_config_001, ds_input.input_data)
        with pytest.raises(ValueError, match="bogus"):
            ds_qc.run_qc_items()

    def test_write_without_run_raises(self, nrtqc_config_001):
        ds_qc = load_nrtqc_step2_qc_dataset(nrtqc_config_001, None)
        with pytest.raises(ValueError, match="qc_data"):
            ds_qc.write_qc_data()


class TestWriteQCData:
    """write_qc_data persists the intermediate flag frame."""

    def test_write(self, qc_step):
        qc_step.write_qc_data()
        assert os.path.exists(qc_step.output_file_name)
        os.remove(qc_step.output_file_name)
