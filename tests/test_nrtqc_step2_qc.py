"""Unit tests for the NRT QC module's step 2 (run QC items).

Runs the full template item set against the CTD test fixture and checks
the produced flag columns: presence, alignment, never-null values within
the flag scheme, and the deferral of the propagation item to step 3.
"""

import os

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
