"""Tests for the deprecation warnings on the NRT QC entry points.

The NRT QC module is moving to ``nrtqclib`` and will be removed from
``aiqclib`` in 0.15.0. ``pytest.ini`` silences these warnings for the rest of
the suite, which exercises the deprecated code on purpose; ``pytest.warns``
records them regardless of that filter.

Each warning must point at the caller (the test file here), so Python's
default filters show it to a user script.
"""

import warnings

import pytest

import aiqclib as aq
from aiqclib.common.config.nrtqc_config import NRTQCConfig
from aiqclib.interface import nrtqc as nrtqc_module

MOVED = "is moving to the nrtqclib package"


def _assert_points_at_this_file(record):
    assert any(MOVED in str(w.message) and w.filename == __file__ for w in record)


def test_run_nrt_qc_warns(monkeypatch):
    def stop(config):
        raise RuntimeError("stop after the warning")

    monkeypatch.setattr(nrtqc_module, "load_nrtqc_step1_input_dataset", stop)
    with pytest.warns(DeprecationWarning, match=MOVED) as record:
        with pytest.raises(RuntimeError, match="stop after the warning"):
            aq.run_nrt_qc(object())
    _assert_points_at_this_file(record)


def test_nrtqc_config_warns():
    with pytest.warns(DeprecationWarning, match=MOVED) as record:
        NRTQCConfig("template:nrt_qc_sets")
    _assert_points_at_this_file(record)


def test_read_config_template_warns():
    with pytest.warns(DeprecationWarning, match="nrtqclib.read_config_template"):
        aq.read_config_template("nrt_qc")


def test_write_and_read_config_warn(tmp_path):
    path = str(tmp_path / "nrt_qc.yaml")
    with pytest.warns(DeprecationWarning, match="nrtqclib.write_config_template"):
        aq.write_config_template(path, "nrt_qc")
    with pytest.warns(DeprecationWarning, match="nrtqclib.read_config") as record:
        aq.read_config(path)
    _assert_points_at_this_file(record)


def test_run_batch_nrt_qc_warns():
    # No config file is given, so the batch stops right after the warning.
    with pytest.warns(DeprecationWarning, match="nrtqclib.run_batch") as record:
        with pytest.raises(ValueError, match="nrt_qc_config"):
            aq.run_batch(mode="nrt_qc")
    _assert_points_at_this_file(record)


@pytest.mark.parametrize("stage", ["prepare", "train", "classify"])
def test_other_stages_do_not_warn(stage, tmp_path):
    path = str(tmp_path / f"{stage}.yaml")
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        aq.write_config_template(path, stage)
        aq.read_config(path)
        aq.read_config_template(stage)
        with pytest.raises(ValueError):
            aq.run_batch(mode=stage)
    assert not any(MOVED in str(w.message) for w in record)
