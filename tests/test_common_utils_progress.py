"""Unit tests for the opt-in progress reporting (``common.utils.progress``).

The four pipeline entry points accept ``verbose=True`` and report their main
steps through :class:`ProgressReporter`. The tests verify that a disabled
reporter is completely silent, that an enabled one names the stage and numbers
its steps, that a failed run is not reported as a finished one, and that all
four entry points expose the option with the same default.

:func:`notice` is the exception to all of that: it is not tied to a reporter or
to ``verbose``, and exists so that a remark a run has to make can be made in
the same shape as the step lines instead of as a warning.
"""

import inspect
import io
import re

import pytest

import aiqclib as aq
from aiqclib.common.utils.progress import (
    NOTICE_WIDTH,
    PREFIX,
    ProgressReporter,
    notice,
    report_progress,
)


class TestProgressReporterDisabled:
    """A disabled reporter must not write anything at all."""

    def test_no_output(self):
        """Every method is a no-op when reporting is off."""
        stream = io.StringIO()
        reporter = ProgressReporter("prepare", 3, enabled=False, stream=stream)
        reporter.step("Reading input data")
        reporter.skip("Comparing flags")
        reporter.finish()
        assert stream.getvalue() == ""

    def test_is_the_default(self):
        """Silence is the default, so existing callers are unaffected."""
        stream = io.StringIO()
        ProgressReporter("prepare", 1, stream=stream).step("Reading input data")
        assert stream.getvalue() == ""


class TestProgressReporterEnabled:
    """An enabled reporter writes one line per step, plus a heading."""

    def test_heading_names_stage_and_label(self):
        """The opening line identifies the workflow and the configuration set."""
        stream = io.StringIO()
        ProgressReporter("classify", 2, enabled=True, label="ds_1", stream=stream)
        assert stream.getvalue().strip() == f"{PREFIX} classify: ds_1"

    def test_heading_without_label(self):
        """A missing set name leaves the stage alone rather than printing None."""
        stream = io.StringIO()
        ProgressReporter("classify", 2, enabled=True, stream=stream)
        assert stream.getvalue().strip() == f"{PREFIX} classify"

    def test_steps_are_numbered_against_the_total(self):
        """Each step line carries its position and the announced total."""
        stream = io.StringIO()
        reporter = ProgressReporter("prepare", 3, enabled=True, stream=stream)
        reporter.step("Reading input data")
        reporter.step("Calculating summary statistics")
        lines = stream.getvalue().splitlines()
        assert "[1/3]" in lines[1] and "Reading input data" in lines[1]
        assert "[2/3]" in lines[2] and "Calculating summary" in lines[2]

    def test_every_line_is_prefixed(self):
        """Output is identifiable as the library's, not stray prints."""
        stream = io.StringIO()
        reporter = ProgressReporter("prepare", 1, enabled=True, stream=stream)
        reporter.step("Reading input data")
        reporter.finish()
        assert all(line.startswith(PREFIX) for line in stream.getvalue().splitlines())

    def test_step_lines_carry_elapsed_seconds(self):
        """The elapsed time is what makes a slow step distinguishable."""
        stream = io.StringIO()
        reporter = ProgressReporter("prepare", 1, enabled=True, stream=stream)
        reporter.step("Reading input data")
        assert re.search(r"\d+\.\d+s", stream.getvalue().splitlines()[1])

    def test_skip_counts_but_is_marked(self):
        """A skipped step keeps the numbering honest and says it was skipped."""
        stream = io.StringIO()
        reporter = ProgressReporter("nrt_qc", 2, enabled=True, stream=stream)
        reporter.step("Running QC items")
        reporter.skip("Comparing flags")
        last = stream.getvalue().splitlines()[-1]
        assert "[2/2]" in last
        assert "(skipped)" in last

    def test_finish_reports_the_step_count_and_total(self):
        """The closing line summarises what actually ran."""
        stream = io.StringIO()
        reporter = ProgressReporter("train", 2, enabled=True, stream=stream)
        reporter.step("Reading training sets")
        reporter.step("Validating models")
        reporter.finish()
        last = stream.getvalue().splitlines()[-1]
        assert last.startswith(f"{PREFIX} train:")
        assert "2 steps in" in last


class TestReportProgressContext:
    """The context manager used by the entry points."""

    def test_closing_line_on_success(self):
        """A completed run ends with the summary line."""
        stream = io.StringIO()
        with report_progress("prepare", 1, enabled=True, stream=stream) as progress:
            progress.step("Reading input data")
        assert "1 steps in" in stream.getvalue().splitlines()[-1]

    def test_no_closing_line_when_the_body_raises(self):
        """A failed run must not look like a finished one."""
        stream = io.StringIO()
        with pytest.raises(RuntimeError):
            with report_progress("prepare", 2, enabled=True, stream=stream) as progress:
                progress.step("Reading input data")
                raise RuntimeError("step failed")
        assert "steps in" not in stream.getvalue()

    def test_exception_propagates_unchanged(self):
        """Reporting must not swallow or reshape a pipeline failure."""
        stream = io.StringIO()
        with pytest.raises(ValueError, match="original message"):
            with report_progress("prepare", 1, enabled=True, stream=stream):
                raise ValueError("original message")


class TestNotice:
    """One-off remarks, wrapped and prefixed like the step lines."""

    LONG = (
        "Computing SHAP values for target 'temp' over 3,671,789 rows. This is "
        "usually the slowest part of a run, and the phase will be quiet for a "
        "long time while it happens."
    )

    @staticmethod
    def _lines(message: str) -> list:
        stream = io.StringIO()
        notice(message, stream=stream)
        return stream.getvalue().splitlines()

    def test_short_message_is_one_line(self):
        """Nothing is wrapped that does not need wrapping."""
        assert self._lines("Short remark.") == [f"{PREFIX} note: Short remark."]

    def test_every_line_carries_the_prefix(self):
        """A wrapped notice must stay as greppable as the step lines."""
        lines = self._lines(self.LONG)
        assert len(lines) > 1
        assert all(line.startswith(PREFIX) for line in lines)

    def test_continuation_lines_align_under_the_text(self):
        """The paragraph reads as one block, indented past the "note:" label."""
        lines = self._lines(self.LONG)
        column = len(f"{PREFIX} note: ")
        for line in lines[1:]:
            assert line[len(PREFIX) : column].isspace()
            assert not line[column].isspace()

    def test_lines_stay_within_the_width(self):
        """A notice must not be the one thing that wraps in the terminal."""
        assert all(len(line) <= NOTICE_WIDTH for line in self._lines(self.LONG))

    def test_a_long_path_is_not_broken_across_lines(self):
        """Paths are the one thing worth overrunning the width for.

        Notices name model and input files. A path split over two lines
        cannot be copied, pasted or grepped for, and the reader has to
        reassemble it by hand to use it at all, so the wrap gives way instead.
        """
        path = "/scratch/workspace/project/data/models/model_temp_xgboost.joblib"
        lines = self._lines(f"The model in '{path}' was written by another version.")

        assert any(path in line for line in lines)

    def test_existing_newlines_are_re_wrapped(self):
        """Callers write a paragraph; the layout is decided here.

        A message built from an f-string across several source lines carries
        no meaningful line breaks of its own, and passing them through would
        produce lines broken at whatever the source happened to be.
        """
        lines = self._lines("A message\nsplit across\nsource lines.")
        assert lines == [f"{PREFIX} note: A message split across source lines."]

    def test_defaults_to_stdout(self, capsys):
        """The remark shares the stream the progress lines use."""
        notice("Written to stdout.")
        assert "Written to stdout." in capsys.readouterr().out


class TestEntryPointSignatures:
    """All four workflows expose the option identically."""

    @pytest.mark.parametrize(
        "function",
        [
            aq.create_training_dataset,
            aq.train_and_evaluate,
            aq.run_nrt_qc,
            aq.classify_dataset,
        ],
        ids=["prepare", "train", "nrt_qc", "classify"],
    )
    def test_verbose_defaults_to_false(self, function):
        """Adding the option must not change what existing callers see."""
        parameter = inspect.signature(function).parameters.get("verbose")
        assert parameter is not None
        assert parameter.default is False
