"""Unit tests for the batch interface (``interface.batch``).

``run_batch`` drives the existing entry points over a table of datasets, so
the tests stub those entry points and check the orchestration: which phases a
mode selects, which set name is selected from which config file, how blank
cells and unknown names are handled, and what the summary frame records.

The table used throughout is the whitespace-aligned form a maintainer is most
likely to write by hand.
"""

import polars as pl
import pytest

from aiqclib.interface import batch as batch_module
from aiqclib.interface.batch import (
    available_modes,
    read_batch_table,
    run_batch,
)


TABLE_TEXT = """\
name       prepare_set_name       training_set_name       classification_set_name
ar_ar      dataset_ar_ar_0001     training_ar_ar_0001     classification_ar_ar_0001
bo_bo      dataset_bo_bo_0001     training_bo_bo_0001     classification_bo_bo_0001
cora_mo    dataset_cora_mo_0001   training_cora_mo_0001   classification_cora_mo_0001
"""


@pytest.fixture
def table_file(tmp_path):
    """The example table as a whitespace-aligned text file."""
    path = tmp_path / "datasets.txt"
    path.write_text(TABLE_TEXT, encoding="utf-8")
    return str(path)


@pytest.fixture
def calls(monkeypatch):
    """Replace the entry points with recorders, returning the call log.

    Each entry is ``(phase, config_file, set_name, verbose)``. ``read_config``
    is stubbed too, so no YAML file has to exist.
    """
    log = []

    def fake_read_config(file_name, set_name=None, auto_select=True):
        return {"config_file": file_name, "set_name": set_name}

    monkeypatch.setattr(batch_module, "read_config", fake_read_config)

    # Phase is frozen, so the recorders go in by rebuilding the tuple.
    monkeypatch.setattr(
        batch_module,
        "PHASES",
        tuple(
            batch_module.Phase(
                phase.name,
                phase.column,
                phase.config_argument,
                _make_recorder(log, phase.name),
            )
            for phase in batch_module.PHASES
        ),
    )
    return log


def _make_recorder(log, phase_name):
    """Build a runner that appends its arguments to ``log``."""

    def recorder(config, verbose=False):
        log.append((phase_name, config["config_file"], config["set_name"], verbose))

    return recorder


ALL_CONFIGS = {
    "prepare_config": "prepare.yaml",
    "training_config": "train.yaml",
    "classification_config": "classify.yaml",
}


class TestReadBatchTable:
    """Reading the dataset table."""

    def test_whitespace_aligned_text(self, table_file):
        """A hand-aligned table splits on runs of whitespace."""
        frame = read_batch_table(table_file)
        assert frame.height == 3
        assert frame.columns[0] == "name"
        assert frame["prepare_set_name"].to_list()[0] == "dataset_ar_ar_0001"

    def test_tsv(self, tmp_path):
        """A .tsv file is read by its delimiter, so values may contain spaces."""
        path = tmp_path / "datasets.tsv"
        path.write_text("name\tprepare_set_name\nar_ar\tdataset_ar_ar_0001\n")
        frame = read_batch_table(str(path))
        assert frame["prepare_set_name"].to_list() == ["dataset_ar_ar_0001"]

    def test_csv(self, tmp_path):
        """A .csv file is read by its delimiter too."""
        path = tmp_path / "datasets.csv"
        path.write_text("name,prepare_set_name\nar_ar,dataset_ar_ar_0001\n")
        frame = read_batch_table(str(path))
        assert frame["prepare_set_name"].to_list() == ["dataset_ar_ar_0001"]

    def test_dataframe_passes_through(self):
        """An in-memory frame skips parsing entirely."""
        frame = read_batch_table(pl.DataFrame({"name": ["ar_ar"], "x": [1]}))
        assert frame["x"].to_list() == ["1"]

    def test_comments_and_blank_lines_ignored(self, tmp_path):
        """A table can be annotated without breaking the parse."""
        path = tmp_path / "datasets.txt"
        path.write_text(
            "# regions\nname  prepare_set_name\n\nar_ar  dataset_ar_ar_0001\n"
        )
        frame = read_batch_table(str(path))
        assert frame.height == 1

    def test_ragged_row_is_reported_with_its_line(self, tmp_path):
        """A row with the wrong field count names the row and the reason."""
        path = tmp_path / "datasets.txt"
        path.write_text("name  prepare_set_name\nar_ar\n")
        with pytest.raises(ValueError, match="row 1"):
            read_batch_table(str(path))

    def test_missing_file(self, tmp_path):
        """A missing table is reported as such, not as a parse error."""
        with pytest.raises(FileNotFoundError, match="does not exist"):
            read_batch_table(str(tmp_path / "nope.txt"))

    def test_header_only_table_is_rejected(self, tmp_path):
        """A table with no data rows is a mistake worth reporting."""
        path = tmp_path / "datasets.txt"
        path.write_text("name  prepare_set_name\n")
        with pytest.raises(ValueError, match="no rows"):
            read_batch_table(str(path))


class TestModes:
    """Which phases a mode selects."""

    def test_available_modes(self):
        """Every phase, plus 'all'."""
        assert available_modes() == ["prepare", "train", "classify", "all"]

    @pytest.mark.parametrize(
        "mode, expected",
        [
            ("prepare", ["prepare"]),
            ("train", ["train"]),
            ("classify", ["classify"]),
            ("all", ["prepare", "train", "classify"]),
        ],
    )
    def test_mode_selects_its_phases(self, mode, expected, table_file, calls):
        """A single-phase mode runs only that phase; 'all' runs them in order."""
        run_batch(table_file, mode=mode, **ALL_CONFIGS)
        assert [call[0] for call in calls[: len(expected)]] == expected
        assert len({call[0] for call in calls}) == len(expected)

    def test_unknown_mode_lists_the_valid_ones(self, table_file, calls):
        """A typo in the mode says what was expected."""
        with pytest.raises(ValueError, match="Expected one of: prepare, train"):
            run_batch(table_file, mode="prepair", **ALL_CONFIGS)

    def test_all_runs_phases_per_dataset(self, table_file, calls):
        """Each dataset completes its phases before the next dataset starts."""
        run_batch(table_file, mode="all", **ALL_CONFIGS)
        assert [call[0] for call in calls[:3]] == ["prepare", "train", "classify"]
        assert [call[2] for call in calls[:3]] == [
            "dataset_ar_ar_0001",
            "training_ar_ar_0001",
            "classification_ar_ar_0001",
        ]


class TestConfigSelection:
    """Each phase reads its own config file and selects its own set name."""

    def test_set_names_come_from_the_matching_column(self, table_file, calls):
        """Row and column together decide which set is selected."""
        run_batch(table_file, mode="prepare", prepare_config="prepare.yaml")
        assert [call[2] for call in calls] == [
            "dataset_ar_ar_0001",
            "dataset_bo_bo_0001",
            "dataset_cora_mo_0001",
        ]

    def test_config_file_per_phase(self, table_file, calls):
        """A phase never reads another phase's configuration file."""
        run_batch(table_file, mode="all", names=["ar_ar"], **ALL_CONFIGS)
        assert [(call[0], call[1]) for call in calls] == [
            ("prepare", "prepare.yaml"),
            ("train", "train.yaml"),
            ("classify", "classify.yaml"),
        ]

    def test_missing_config_names_the_argument(self, table_file, calls):
        """Running a phase without its config says which argument to pass."""
        with pytest.raises(ValueError, match="training_config"):
            run_batch(table_file, mode="train")

    def test_missing_column_names_the_column(self, tmp_path, calls):
        """A mode whose column is absent is reported before anything runs."""
        path = tmp_path / "datasets.txt"
        path.write_text("name  prepare_set_name\nar_ar  dataset_ar_ar_0001\n")
        with pytest.raises(ValueError, match="training_set_name"):
            run_batch(str(path), mode="train", training_config="train.yaml")
        assert calls == []

    def test_verbose_is_passed_through(self, table_file, calls, capsys):
        """The entry points get the flag, and the batch reports its own lines."""
        run_batch(
            table_file, mode="prepare", prepare_config="prepare.yaml", verbose=True
        )
        assert all(call[3] is True for call in calls)
        out = capsys.readouterr().out
        assert "[aiqclib] batch:" in out
        assert "ar_ar / prepare" in out


class TestRowSelection:
    """Choosing which datasets to run."""

    def test_names_restricts_the_run(self, table_file, calls):
        """Only the requested datasets run."""
        run_batch(
            table_file,
            mode="prepare",
            prepare_config="prepare.yaml",
            names=["bo_bo", "cora_mo"],
        )
        assert [call[2] for call in calls] == [
            "dataset_bo_bo_0001",
            "dataset_cora_mo_0001",
        ]

    def test_unknown_name_lists_the_available_ones(self, table_file, calls):
        """A typo in a name fails before any work is done."""
        with pytest.raises(ValueError, match="Available: ar_ar, bo_bo, cora_mo"):
            run_batch(
                table_file,
                mode="prepare",
                prepare_config="prepare.yaml",
                names=["ar_gl"],
            )
        assert calls == []

    def test_blank_cell_skips_that_phase(self, tmp_path, calls):
        """A dataset with no set name for a phase skips it, and says so."""
        path = tmp_path / "datasets.tsv"
        path.write_text("name\tprepare_set_name\nar_ar\tdataset_ar_ar_0001\nbo_bo\t\n")
        summary = run_batch(str(path), mode="prepare", prepare_config="prepare.yaml")
        assert [call[2] for call in calls] == ["dataset_ar_ar_0001"]
        assert summary["status"].to_list() == ["ok", "skipped"]

    def test_first_column_is_used_when_not_called_name(self, tmp_path, calls):
        """A table whose identifier column has another header still works."""
        path = tmp_path / "datasets.tsv"
        path.write_text("name_name\tprepare_set_name\nar_ar\tdataset_ar_ar_0001\n")
        summary = run_batch(str(path), mode="prepare", prepare_config="prepare.yaml")
        assert summary["name"].to_list() == ["ar_ar"]


class TestWithoutATable:
    """Running with no table at all, letting each config select its own set."""

    def test_each_phase_runs_once_without_a_set_name(self, calls):
        """No table means one run per phase, with no set named."""
        run_batch(mode="all", **ALL_CONFIGS)
        assert [call[0] for call in calls] == ["prepare", "train", "classify"]
        assert [call[2] for call in calls] == [None, None, None]

    def test_single_mode_without_a_table(self, calls):
        """A single phase works the same way."""
        run_batch(mode="prepare", prepare_config="prepare.yaml")
        assert [(call[0], call[1]) for call in calls] == [("prepare", "prepare.yaml")]

    def test_summary_has_no_dataset_name(self, calls):
        """There is no name to record, so the column is null rather than invented."""
        summary = run_batch(mode="prepare", prepare_config="prepare.yaml")
        assert summary.height == 1
        assert summary["name"].to_list() == [None]
        assert summary["status"].to_list() == ["ok"]

    def test_summary_records_the_set_the_config_chose(self, monkeypatch, calls):
        """The summary says which set ran, even though the caller named none."""

        class FakeConfig(dict):
            dataset_name = "auto_selected_0001"

        def fake_read_config(file_name, set_name=None, auto_select=True):
            config = FakeConfig(config_file=file_name, set_name=set_name)
            return config

        monkeypatch.setattr(batch_module, "read_config", fake_read_config)
        summary = run_batch(mode="prepare", prepare_config="prepare.yaml")
        assert summary["set_name"].to_list() == ["auto_selected_0001"]

    def test_missing_config_is_still_reported(self, calls):
        """Dropping the table does not drop the config requirement."""
        with pytest.raises(ValueError, match="prepare_config"):
            run_batch(mode="prepare")

    def test_names_without_a_table_is_rejected(self, calls):
        """'names' selects table rows, so it cannot apply without a table."""
        with pytest.raises(ValueError, match="no table was given"):
            run_batch(mode="prepare", prepare_config="prepare.yaml", names=["ar_ar"])

    def test_verbose_reports_the_runs(self, calls, capsys):
        """The reporting says the config file is choosing the set."""
        run_batch(mode="prepare", prepare_config="prepare.yaml", verbose=True)
        out = capsys.readouterr().out
        assert "set chosen by the config file" in out


class TestSummary:
    """The frame describing what ran."""

    def test_one_row_per_dataset_and_phase(self, table_file, calls):
        """Three datasets over three phases give nine rows."""
        summary = run_batch(table_file, mode="all", **ALL_CONFIGS)
        assert summary.height == 9
        assert summary.columns == list(batch_module.SUMMARY_SCHEMA)

    def test_successful_runs_are_recorded_ok(self, table_file, calls):
        """A completed phase records its set name and a duration."""
        summary = run_batch(table_file, mode="prepare", prepare_config="prepare.yaml")
        assert summary["status"].to_list() == ["ok", "ok", "ok"]
        assert summary["error"].to_list() == [None, None, None]
        assert all(seconds >= 0.0 for seconds in summary["seconds"].to_list())


class TestFailures:
    """What happens when one dataset fails."""

    @pytest.fixture
    def failing_calls(self, monkeypatch):
        """Entry points where bo_bo raises and the others succeed."""
        log = []

        def fake_read_config(file_name, set_name=None, auto_select=True):
            return {"config_file": file_name, "set_name": set_name}

        monkeypatch.setattr(batch_module, "read_config", fake_read_config)

        def runner(config, verbose=False):
            if "bo_bo" in config["set_name"]:
                raise RuntimeError("bo_bo is broken")
            log.append(config["set_name"])

        monkeypatch.setattr(
            batch_module,
            "PHASES",
            tuple(
                batch_module.Phase(
                    phase.name, phase.column, phase.config_argument, runner
                )
                for phase in batch_module.PHASES
            ),
        )
        return log

    def test_raises_by_default(self, table_file, failing_calls):
        """A failure stops the batch, so it cannot pass unnoticed."""
        with pytest.raises(RuntimeError, match="bo_bo is broken"):
            run_batch(table_file, mode="prepare", prepare_config="prepare.yaml")
        assert failing_calls == ["dataset_ar_ar_0001"]

    def test_continue_on_error_records_and_carries_on(self, table_file, failing_calls):
        """With the flag set, later datasets still run and the failure is kept."""
        summary = run_batch(
            table_file,
            mode="prepare",
            prepare_config="prepare.yaml",
            continue_on_error=True,
        )
        assert summary["status"].to_list() == ["ok", "failed", "ok"]
        assert failing_calls == ["dataset_ar_ar_0001", "dataset_cora_mo_0001"]

    def test_recorded_error_keeps_type_and_message(self, table_file, failing_calls):
        """The summary says what went wrong, not just that something did."""
        summary = run_batch(
            table_file,
            mode="prepare",
            prepare_config="prepare.yaml",
            continue_on_error=True,
        )
        error = summary.filter(pl.col("status") == "failed")["error"].to_list()[0]
        assert error == "RuntimeError: bo_bo is broken"
