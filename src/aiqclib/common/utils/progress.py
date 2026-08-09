"""
Opt-in progress reporting for the pipeline entry points.

The four workflows (``create_training_dataset``, ``train_and_evaluate``,
``run_nrt_qc``, ``classify_dataset``) run for minutes on real datasets while
printing nothing, so there is no way to tell a slow step from a hung one. Each
accepts ``verbose=True`` to report its main steps through this module.

Output goes to stdout, one line per step as the step begins, so it appears
while the work is running rather than at the end:

.. code-block:: text

   [aiqclib] classify: dataset_0001
   [aiqclib]   [1/7]    0.0s  Reading input data
   [aiqclib]   [2/7]    4.1s  Calculating summary statistics
   [aiqclib]   [3/7]   11.7s  Selecting profiles
   [aiqclib] classify: 7 steps in 38.2s

The time on each line is the elapsed time of the run so far, so the cost of a
step is the gap to the next line, and the final line gives the total.
"""

import sys
from contextlib import contextmanager
from time import perf_counter
from typing import Iterator, Optional, TextIO

#: Prefix identifying the library on every reported line.
PREFIX: str = "[aiqclib]"


class ProgressReporter:
    """
    Report the main steps of a workflow, or stay silent.

    A disabled reporter accepts the same calls and writes nothing, so the entry
    points read the same whether or not reporting is on.

    :ivar stage: The workflow name shown on the opening and closing lines.
    :vartype stage: str
    :ivar total_steps: How many steps the workflow will report.
    :vartype total_steps: int
    :ivar enabled: Whether anything is written at all.
    :vartype enabled: bool
    """

    def __init__(
        self,
        stage: str,
        total_steps: int,
        enabled: bool = False,
        label: Optional[str] = None,
        stream: Optional[TextIO] = None,
    ) -> None:
        """
        Initialize the reporter and, when enabled, write the opening line.

        :param stage: The workflow name, e.g. ``"classify"``.
        :type stage: str
        :param total_steps: The number of steps that will be reported.
        :type total_steps: int
        :param enabled: When False every method is a no-op. Defaults to False.
        :type enabled: bool
        :param label: The configuration set being processed, shown alongside
                      the stage when known.
        :type label: Optional[str]
        :param stream: Where to write. Defaults to ``sys.stdout``.
        :type stream: Optional[TextIO]
        """
        self.stage: str = stage
        self.total_steps: int = total_steps
        self.enabled: bool = enabled
        self._stream: TextIO = stream if stream is not None else sys.stdout
        self._index: int = 0
        self._start: float = perf_counter()

        if self.enabled:
            heading = f"{stage}: {label}" if label else stage
            self._write(f"{PREFIX} {heading}")

    def _write(self, line: str) -> None:
        """
        Write one line, flushing so progress appears while the run continues.

        :param line: The text to write.
        :type line: str
        :return: None
        :rtype: None
        """
        print(line, file=self._stream, flush=True)

    @property
    def elapsed(self) -> float:
        """
        Seconds since the reporter was created.

        :return: The elapsed time of the run so far.
        :rtype: float
        """
        return perf_counter() - self._start

    def step(self, description: str) -> None:
        """
        Report that a step is starting.

        :param description: What the step does, e.g. ``"Reading input data"``.
        :type description: str
        :return: None
        :rtype: None
        """
        self._index += 1
        if not self.enabled:
            return
        self._write(
            f"{PREFIX}   [{self._index}/{self.total_steps}] "
            f"{self.elapsed:6.1f}s  {description}"
        )

    def skip(self, description: str) -> None:
        """
        Report a step that the configuration does not require.

        The step still consumes its number, so the count keeps matching the
        total announced at the start.

        :param description: What the skipped step would have done.
        :type description: str
        :return: None
        :rtype: None
        """
        self._index += 1
        if not self.enabled:
            return
        self._write(
            f"{PREFIX}   [{self._index}/{self.total_steps}] "
            f"{self.elapsed:6.1f}s  {description} (skipped)"
        )

    def finish(self) -> None:
        """
        Report that the workflow is done, with the total elapsed time.

        :return: None
        :rtype: None
        """
        if not self.enabled:
            return
        self._write(
            f"{PREFIX} {self.stage}: {self._index} steps in {self.elapsed:.1f}s"
        )


@contextmanager
def report_progress(
    stage: str,
    total_steps: int,
    enabled: bool = False,
    label: Optional[str] = None,
    stream: Optional[TextIO] = None,
) -> Iterator[ProgressReporter]:
    """
    Run a workflow with a reporter, writing the closing line on success.

    The closing line is skipped when the body raises, so a failed run is not
    reported as a completed one; the exception propagates unchanged.

    :param stage: The workflow name, e.g. ``"classify"``.
    :type stage: str
    :param total_steps: The number of steps that will be reported.
    :type total_steps: int
    :param enabled: When False nothing is written. Defaults to False.
    :type enabled: bool
    :param label: The configuration set being processed, when known.
    :type label: Optional[str]
    :param stream: Where to write. Defaults to ``sys.stdout``.
    :type stream: Optional[TextIO]
    :yields: The reporter to call :meth:`ProgressReporter.step` on.
    :rtype: Iterator[ProgressReporter]
    """
    reporter = ProgressReporter(
        stage=stage,
        total_steps=total_steps,
        enabled=enabled,
        label=label,
        stream=stream,
    )
    yield reporter
    reporter.finish()
