"""
Batch orchestration across several datasets.

Running the same workflow over a handful of regions means repeating the same
three calls per dataset, differing only in which named set each config selects.
:func:`run_batch` takes that repetition: a table naming the configuration sets
per dataset, a mode saying which phases to run, and one config file per phase.

The table is a delimited text file whose first column names the dataset and
whose remaining columns give the set name to select for each phase:

.. code-block:: text

   name       prepare_set_name      training_set_name      classification_set_name
   ar_ar      dataset_ar_ar_0001    training_ar_ar_0001    classification_ar_ar_0001
   bo_bo      dataset_bo_bo_0001    training_bo_bo_0001    classification_bo_bo_0001

Only the columns of the phases being run have to be present, so a table used
for ``mode="prepare"`` needs nothing but the name and prepare columns. A blank
cell skips that phase for that dataset.

Every run returns a summary frame, one row per dataset and phase, recording
whether it succeeded and how long it took. New phases are added by extending
:data:`PHASES`; nothing else in the module is phase-specific.
"""

import os
from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import polars as pl

from aiqclib.common.base.config_base import ConfigBase
from aiqclib.common.utils.file import expand_path
from aiqclib.common.utils.progress import PREFIX
from aiqclib.interface.classify import classify_dataset
from aiqclib.interface.config import read_config
from aiqclib.interface.prepare import create_training_dataset
from aiqclib.interface.train import train_and_evaluate


@dataclass(frozen=True)
class Phase:
    """
    One workflow that :func:`run_batch` can run for a dataset.

    :ivar name: The mode name a caller passes, e.g. ``"prepare"``.
    :vartype name: str
    :ivar column: The table column holding the set name for this phase.
    :vartype column: str
    :ivar config_argument: The :func:`run_batch` keyword supplying its config file.
    :vartype config_argument: str
    :ivar runner: The entry point to call with the selected configuration.
    :vartype runner: Callable[[ConfigBase, bool], None]
    """

    name: str
    column: str
    config_argument: str
    runner: Callable[[ConfigBase, bool], None]


#: The phases a batch can run, in the order ``mode="all"`` runs them.
PHASES: Tuple[Phase, ...] = (
    Phase("prepare", "prepare_set_name", "prepare_config", create_training_dataset),
    Phase("train", "training_set_name", "training_config", train_and_evaluate),
    Phase(
        "classify",
        "classification_set_name",
        "classification_config",
        classify_dataset,
    ),
)

#: The mode selecting every phase.
ALL_MODE: str = "all"

#: Shown in place of a dataset name when the batch runs without a table.
AUTO_LABEL: str = "(no table)"

#: Column schema of the summary frame returned by :func:`run_batch`.
SUMMARY_SCHEMA: Dict = {
    "name": pl.Utf8,
    "phase": pl.Utf8,
    "set_name": pl.Utf8,
    "status": pl.Utf8,
    "seconds": pl.Float64,
    "error": pl.Utf8,
}


def available_modes() -> List[str]:
    """
    Return the accepted ``mode`` values.

    :return: Each phase name, followed by ``"all"``.
    :rtype: List[str]
    """
    return [phase.name for phase in PHASES] + [ALL_MODE]


def _resolve_phases(mode: str) -> Tuple[Phase, ...]:
    """
    Map a mode onto the phases it runs.

    :param mode: A phase name or ``"all"``.
    :type mode: str
    :raises ValueError: If the mode is not one of :func:`available_modes`.
    :return: The phases to run, in order.
    :rtype: Tuple[Phase, ...]
    """
    if mode == ALL_MODE:
        return PHASES
    for phase in PHASES:
        if phase.name == mode:
            return (phase,)
    raise ValueError(
        f"Unknown mode '{mode}'. Expected one of: {', '.join(available_modes())}."
    )


def read_batch_table(table: Union[str, pl.DataFrame]) -> pl.DataFrame:
    """
    Read the dataset table, accepting a path or an in-memory frame.

    ``.tsv`` and ``.csv`` files (optionally gzipped) are read by their
    delimiter; any other extension is treated as whitespace-separated, which is
    what a hand-aligned table pasted into a text file usually is.

    :param table: Path to the table, or an already-loaded DataFrame.
    :type table: Union[str, polars.DataFrame]
    :raises FileNotFoundError: If the path does not exist.
    :raises ValueError: If the table has no rows or no columns.
    :return: The table, with every column read as text.
    :rtype: polars.DataFrame
    """
    if isinstance(table, pl.DataFrame):
        frame = table.select(pl.all().cast(pl.Utf8))
    else:
        table = expand_path(table)
        if not os.path.exists(table):
            raise FileNotFoundError(f"Batch table '{table}' does not exist.")

        name = os.path.basename(table).lower()
        if name.endswith((".csv", ".csv.gz")):
            separator = ","
        elif name.endswith((".tsv", ".tsv.gz")):
            separator = "\t"
        else:
            separator = None

        if separator is None:
            frame = _read_whitespace_table(table)
        else:
            frame = pl.read_csv(
                table,
                separator=separator,
                infer_schema_length=0,
            )

    frame = frame.with_columns(pl.all().str.strip_chars())
    if frame.width == 0:
        raise ValueError("Batch table has no columns.")
    if frame.height == 0:
        raise ValueError("Batch table has no rows.")
    return frame


def _read_whitespace_table(table: str) -> pl.DataFrame:
    """
    Read a table whose columns are separated by runs of spaces or tabs.

    Blank lines and ``#`` comment lines are ignored, so a table can be
    annotated. Every row must have as many fields as the header.

    :param table: Path to the table file.
    :type table: str
    :raises ValueError: If the file holds no header, or a row's field count
                        does not match the header's.
    :return: The parsed table, every column as text.
    :rtype: polars.DataFrame
    """
    with open(table, "r", encoding="utf-8") as handle:
        rows = [
            line.split()
            for line in handle
            if line.strip() and not line.lstrip().startswith("#")
        ]

    if not rows:
        raise ValueError(f"Batch table '{table}' has no header row.")

    header, *records = rows
    for number, record in enumerate(records, start=1):
        if len(record) != len(header):
            raise ValueError(
                f"Batch table '{table}', row {number}: found {len(record)} "
                f"fields but the header has {len(header)} "
                f"({', '.join(header)}). Whitespace-separated tables cannot "
                f"hold values containing spaces; use a TSV or CSV instead."
            )

    return pl.DataFrame(
        {
            column: [record[index] for record in records]
            for index, column in enumerate(header)
        },
        schema={column: pl.Utf8 for column in header},
    )


def _name_column(frame: pl.DataFrame) -> str:
    """
    Find the column naming each dataset.

    A column literally called ``name`` wins; otherwise the first column is
    used, so a table whose identifier column is called something else still
    works without renaming.

    :param frame: The batch table.
    :type frame: polars.DataFrame
    :return: The column holding the dataset names.
    :rtype: str
    """
    for column in frame.columns:
        if column.lower() == "name":
            return column
    return frame.columns[0]


def _validate_columns(frame: pl.DataFrame, phases: Sequence[Phase]) -> None:
    """
    Check the table carries a column for every phase being run.

    :param frame: The batch table.
    :type frame: polars.DataFrame
    :param phases: The phases the run needs columns for.
    :type phases: Sequence[Phase]
    :raises ValueError: If a required column is missing.
    :return: None
    :rtype: None
    """
    missing = [phase.column for phase in phases if phase.column not in frame.columns]
    if missing:
        raise ValueError(
            f"Batch table is missing column(s) required by this mode: "
            f"{', '.join(missing)}. Found: {', '.join(frame.columns)}."
        )


def _resolve_config_files(
    phases: Sequence[Phase], config_files: Dict[str, Optional[str]]
) -> None:
    """
    Check a config file was supplied for every phase being run.

    :param phases: The phases the run needs config files for.
    :type phases: Sequence[Phase]
    :param config_files: The config arguments as passed to :func:`run_batch`.
    :type config_files: Dict[str, Optional[str]]
    :raises ValueError: If a phase has no config file.
    :return: None
    :rtype: None
    """
    missing = [
        phase.config_argument
        for phase in phases
        if not config_files.get(phase.config_argument)
    ]
    if missing:
        raise ValueError(
            f"This mode needs the config file(s): {', '.join(missing)}. "
            f"Pass them to run_batch, e.g. run_batch(..., {missing[0]}='...')."
        )


def _select_rows(
    frame: pl.DataFrame, name_column: str, names: Optional[Sequence[str]]
) -> pl.DataFrame:
    """
    Restrict the table to the requested dataset names.

    :param frame: The batch table.
    :type frame: polars.DataFrame
    :param name_column: The column holding the dataset names.
    :type name_column: str
    :param names: The names to keep, or None to keep every row.
    :type names: Optional[Sequence[str]]
    :raises ValueError: If a requested name is not in the table.
    :return: The rows to run.
    :rtype: polars.DataFrame
    """
    if names is None:
        return frame

    wanted = list(names)
    known = set(frame[name_column].to_list())
    unknown = [name for name in wanted if name not in known]
    if unknown:
        raise ValueError(
            f"Name(s) not found in the batch table: {', '.join(unknown)}. "
            f"Available: {', '.join(sorted(known))}."
        )
    return frame.filter(pl.col(name_column).is_in(wanted))


def run_batch(
    table: Optional[Union[str, pl.DataFrame]] = None,
    mode: str = ALL_MODE,
    prepare_config: Optional[str] = None,
    training_config: Optional[str] = None,
    classification_config: Optional[str] = None,
    names: Optional[Sequence[str]] = None,
    verbose: bool = False,
    continue_on_error: bool = False,
) -> pl.DataFrame:
    """
    Run one or more workflows over every dataset named in a table.

    For each row, the set name in the phase's column is selected from that
    phase's configuration file, and the corresponding entry point is called.
    A blank cell skips that phase for that dataset.

    Without a table the batch runs each phase once with no set name, leaving
    each configuration file to select its own set. That is the whole batch for
    a project whose config files hold a single set each, and it turns
    ``run_batch`` into a way of running the phases in order.

    :param table: Path to the dataset table, or an already-loaded DataFrame.
                  The first column (or one called ``name``) identifies the
                  dataset; the phase columns give the set name to select.
                  ``None`` runs each phase once without naming a set.
    :type table: Optional[Union[str, polars.DataFrame]]
    :param mode: Which phases to run: a phase name from :func:`available_modes`
                 or ``"all"``. Defaults to ``"all"``.
    :type mode: str
    :param prepare_config: The configuration file for the prepare phase.
    :type prepare_config: Optional[str]
    :param training_config: The configuration file for the train phase.
    :type training_config: Optional[str]
    :param classification_config: The configuration file for the classify phase.
    :type classification_config: Optional[str]
    :param names: Run only these datasets, instead of every row. Requires a
                  table, since there are no names to choose from without one.
    :type names: Optional[Sequence[str]]
    :param verbose: Report each dataset and phase as it starts, and pass the
                    flag on to the entry points. Defaults to False.
    :type verbose: bool
    :param continue_on_error: When True, a failing dataset is recorded and the
                              batch carries on; when False the exception
                              propagates immediately. Defaults to False.
    :type continue_on_error: bool
    :raises ValueError: If the mode is unknown, the table lacks a column or a
                        config file needed by the mode, a requested name is not
                        in the table, or ``names`` is given without a table.
    :return: One row per dataset and phase (see :data:`SUMMARY_SCHEMA`), with
             ``status`` one of ``"ok"``, ``"skipped"`` or ``"failed"``.
    :rtype: polars.DataFrame

    :Example:

    .. code-block:: python

        import aiqclib as aq

        summary = aq.run_batch(
            "datasets.tsv",
            mode="all",
            prepare_config="prepare.yaml",
            training_config="train.yaml",
            classification_config="classify.yaml",
            verbose=True,
        )
        print(summary.filter(pl.col("status") == "failed"))
    """
    phases = _resolve_phases(mode)
    config_files = {
        "prepare_config": prepare_config,
        "training_config": training_config,
        "classification_config": classification_config,
    }
    _resolve_config_files(phases, config_files)

    if table is None:
        if names is not None:
            raise ValueError(
                "'names' selects rows of a batch table, but no table was given. "
                "Pass a table, or drop 'names' to let each config select its "
                "own set."
            )
        # One unnamed dataset whose set names are left to the config files.
        records: List[Dict] = [{phase.column: None for phase in phases}]
        name_column = None
    else:
        frame = read_batch_table(table)
        _validate_columns(frame, phases)
        name_column = _name_column(frame)
        frame = _select_rows(frame, name_column, names)
        records = list(frame.iter_rows(named=True))

    started = perf_counter()
    if verbose:
        scope = f"{len(records)} datasets" if name_column else "the configured sets"
        print(
            f"{PREFIX} batch: {scope} x {len(phases)} "
            f"phase(s) [{', '.join(phase.name for phase in phases)}]",
            flush=True,
        )

    rows: List[Dict] = []
    for record in records:
        dataset_name = record[name_column] if name_column else None
        for phase in phases:
            rows.append(
                _run_one(
                    phase=phase,
                    dataset_name=dataset_name,
                    set_name=record.get(phase.column),
                    config_file=config_files[phase.config_argument],
                    auto_select=name_column is None,
                    verbose=verbose,
                    continue_on_error=continue_on_error,
                )
            )

    summary = pl.DataFrame(rows, schema=SUMMARY_SCHEMA)
    if verbose:
        counts = dict(
            summary.group_by("status").len().iter_rows()  # status -> count
        )
        detail = ", ".join(
            f"{count} {status}" for status, count in sorted(counts.items())
        )
        print(
            f"{PREFIX} batch: {summary.height} runs in "
            f"{perf_counter() - started:.1f}s ({detail})",
            flush=True,
        )
    return summary


def _run_one(
    phase: Phase,
    dataset_name: Optional[str],
    set_name: Optional[str],
    config_file: str,
    auto_select: bool,
    verbose: bool,
    continue_on_error: bool,
) -> Dict:
    """
    Run a single phase for a single dataset and describe the outcome.

    :param phase: The phase to run.
    :type phase: Phase
    :param dataset_name: The dataset's name in the table, or None when the
                         batch is running without one.
    :type dataset_name: Optional[str]
    :param set_name: The configuration set to select. Blank skips the phase,
                     unless ``auto_select`` leaves the choice to the config.
    :type set_name: Optional[str]
    :param config_file: The phase's configuration file.
    :type config_file: str
    :param auto_select: When True, no set is named and the configuration file
                        selects its own; a blank ``set_name`` is then not a skip.
    :type auto_select: bool
    :param verbose: Whether to report the run and pass the flag on.
    :type verbose: bool
    :param continue_on_error: Whether a failure is recorded instead of raised.
    :type continue_on_error: bool
    :raises Exception: Whatever the entry point raised, when
                       ``continue_on_error`` is False.
    :return: One summary row (see :data:`SUMMARY_SCHEMA`).
    :rtype: Dict
    """
    label = dataset_name or AUTO_LABEL

    if not set_name and not auto_select:
        if verbose:
            print(
                f"{PREFIX} batch: {label} / {phase.name} (skipped, no {phase.column})",
                flush=True,
            )
        return {
            "name": dataset_name,
            "phase": phase.name,
            "set_name": None,
            "status": "skipped",
            "seconds": 0.0,
            "error": None,
        }

    if verbose:
        print(
            f"{PREFIX} batch: {label} / {phase.name} "
            f"({set_name or 'set chosen by the config file'})",
            flush=True,
        )

    started = perf_counter()
    try:
        if auto_select:
            config = read_config(config_file)
            # Record what the config file actually selected, so the summary
            # says which set ran even though the caller did not name one.
            set_name = getattr(config, "dataset_name", None)
        else:
            config = read_config(config_file, set_name=set_name)
        phase.runner(config, verbose=verbose)
    except Exception as error:  # noqa: BLE001 - recorded or re-raised below
        if not continue_on_error:
            raise
        if verbose:
            print(
                f"{PREFIX} batch: {dataset_name} / {phase.name} FAILED: {error}",
                flush=True,
            )
        return {
            "name": dataset_name,
            "phase": phase.name,
            "set_name": set_name,
            "status": "failed",
            "seconds": perf_counter() - started,
            "error": f"{type(error).__name__}: {error}",
        }

    return {
        "name": dataset_name,
        "phase": phase.name,
        "set_name": set_name,
        "status": "ok",
        "seconds": perf_counter() - started,
        "error": None,
    }
