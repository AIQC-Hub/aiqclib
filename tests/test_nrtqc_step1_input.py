"""Unit tests for the NRT QC module's step 1 (read input)."""

import pytest

from aiqclib.common.loader.nrtqc_loader import load_nrtqc_step1_input_dataset
from aiqclib.common.utils.input_validation import REQUIRED_INPUT_COLUMNS
from aiqclib.nrtqc.step1_read_input.dataset_all import InputDataSetAll


class TestNRTQCInput:
    """Step 1 reads and validates the input parquet."""

    def test_loader_returns_configured_class(self, nrtqc_config_001):
        ds = load_nrtqc_step1_input_dataset(nrtqc_config_001)
        assert isinstance(ds, InputDataSetAll)

    def test_input_file_name_resolution(self, nrtqc_config_001, input_dir):
        ds = load_nrtqc_step1_input_dataset(nrtqc_config_001)
        assert ds.input_file_name == str(input_dir / "nrt_cora_bo_test.parquet")

    def test_read_input_data(self, nrtqc_config_001):
        """The fixture parquet loads with all mandatory columns."""
        ds = load_nrtqc_step1_input_dataset(nrtqc_config_001)
        ds.read_input_data()

        assert ds.input_data.shape[0] == 3267
        assert set(REQUIRED_INPUT_COLUMNS) <= set(ds.input_data.columns)
        # The variables and existing flags used later in the pipeline.
        assert {"temp", "psal", "temp_qc", "psal_qc"} <= set(ds.input_data.columns)

    def test_class_mismatch_raises(self, nrtqc_config_001):
        """A config naming another class for 'input' is rejected."""
        nrtqc_config_001.set_base_class("input", "SomethingElse")
        with pytest.raises(ValueError, match="Unknown NRT QC class"):
            load_nrtqc_step1_input_dataset(nrtqc_config_001)
