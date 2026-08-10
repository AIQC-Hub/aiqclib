"""
Diagnostics for datasets that are structurally fine but carry no information.

Some inputs produce metrics that look perfect while measuring nothing. The
clearest case is an evaluation set whose labels are all the same class: every
prediction of that class is correct, so precision, recall, f1 and accuracy all
come out as 1.0, and no ROC or precision-recall curve can be drawn at all. The
numbers are not wrong, they are meaningless, and nothing in the output says so.

A dataset with no rows at all is the more severe version of the same problem,
and it does not stay quiet: features built by pivoting (the flanking values)
take their column names from the values actually present, so an empty dataset
produces no such columns, and the failure surfaces much later as a mismatch
between the model's feature names and the input's. That message describes a
symptom several steps removed from the cause.

Each is reported here at the point it is detected, naming the target and what
to check. Whether it is an error or a warning depends on what survives it:

* No rows at all is an error — nothing downstream can recover from it.
* Single-class labels are an error when a model is about to be *fitted* on
  them, because the resulting model predicts one class at one constant score
  and the model file gives no sign of it afterwards.
* Single-class labels are a warning when a model is merely being *evaluated*
  against them: the metrics are degenerate but the model is not, and applying
  a model to data that happens to be entirely good is a legitimate QC run.
"""

import warnings
from typing import Optional

import polars as pl


def _context(target_name: Optional[str], k: int) -> str:
    """
    Build the "which evaluation" prefix of a diagnostic message.

    :param target_name: The target being evaluated, when known.
    :type target_name: Optional[str]
    :param k: The fold number; 0 means a single, unfolded evaluation.
    :type k: int
    :return: A phrase naming the evaluation, e.g. ``"target 'temp', fold 3"``.
    :rtype: str
    """
    parts = [f"target '{target_name}'" if target_name else "this target"]
    if k:
        parts.append(f"fold {k}")
    return ", ".join(parts)


def check_dataset_not_empty(
    frame: pl.DataFrame,
    dataset_name: str,
    target_name: Optional[str] = None,
    k: int = 0,
) -> None:
    """
    Raise when a dataset that is about to be written or modelled has no rows.

    An empty dataset cannot be trained on, evaluated or classified, and it is
    not detected as such downstream: the pivot-built flanking features emit no
    columns for it, so the run fails later with a feature-name mismatch that
    names 30 missing columns instead of the empty input behind them. Failing
    here names the dataset and the usual cause instead.

    :param frame: The dataset to check.
    :type frame: polars.DataFrame
    :param dataset_name: What the dataset is, used in the message, e.g.
                         ``"training set"`` or ``"classification input"``.
    :type dataset_name: str
    :param target_name: The target the dataset belongs to, when known.
    :type target_name: Optional[str]
    :param k: The fold number; 0 means the dataset is not a fold.
    :type k: int
    :raises ValueError: If ``frame`` has no rows.
    """
    if frame.height:
        return

    where = _context(target_name, k)
    raise ValueError(
        f"The {dataset_name} for {where} has no rows, so there is nothing to "
        f"train on, evaluate or classify. This usually means the row filters "
        f"removed everything — check 'keep_years' / 'remove_years' against the "
        f"years the input actually covers — or that no profile matched the "
        f"selection criteria. Left unchecked this surfaces much later as a "
        f"mismatch between the model's feature names and the input's."
    )


def check_labels_not_single_class(
    labels: pl.Series,
    target_name: Optional[str] = None,
    k: int = 0,
) -> None:
    """
    Raise when the labels a model is about to be fitted on hold one class.

    Fitting succeeds on single-class labels and produces a model that predicts
    that class for every row, with a positive-class score that has exactly one
    distinct value — so no ``prediction_threshold`` can recover it. The model
    is written out and used at classification time looking like any other,
    which makes this worth refusing rather than warning about.

    Evaluating on single-class labels is a lesser problem — the scores are
    degenerate but the model is not — and stays a warning; see
    :func:`warn_single_class_labels`.

    :param labels: The labels the model is about to be trained on.
    :type labels: polars.Series
    :param target_name: The target being trained, used in the message.
    :type target_name: Optional[str]
    :param k: The fold number; 0 means a single, unfolded fit.
    :type k: int
    :raises ValueError: If fewer than two classes are present.
    """
    present = labels.drop_nulls().unique().sort().to_list()
    if len(present) > 1:
        return

    where = _context(target_name, k)
    found = f"all label={present[0]}" if present else "no labelled rows at all"
    raise ValueError(
        f"Training data for {where} has {found} ({labels.len()} rows), so a "
        f"model fitted on it would predict that one class for everything, at a "
        f"single constant score that no 'prediction_threshold' can separate. "
        f"Check 'pos_flag_values' / 'neg_flag_values' against the flag values "
        f"actually present in the input; if this variable genuinely has no "
        f"flagged observations, drop it from the 'target_set' rather than "
        f"training a model that cannot flag anything."
    )


def warn_single_class_labels(
    labels: pl.Series,
    target_name: Optional[str] = None,
    k: int = 0,
) -> bool:
    """
    Warn when an evaluation's labels hold fewer than two classes.

    A single-class evaluation set yields a report full of 1.0 scores that
    measures nothing, and leaves the ROC and precision-recall curves empty.
    Both are reported here as one warning rather than being left to surface as
    an unexplained plotting warning further downstream.

    :param labels: The true labels the evaluation was scored against.
    :type labels: polars.Series
    :param target_name: The target being evaluated, used in the message.
    :type target_name: Optional[str]
    :param k: The fold number; 0 means a single, unfolded evaluation.
    :type k: int
    :return: True when the labels are degenerate (fewer than two classes).
    :rtype: bool
    """
    present = labels.drop_nulls().unique().sort().to_list()
    if len(present) > 1:
        return False

    where = _context(target_name, k)
    if not present:
        warnings.warn(
            f"Evaluation for {where} has no labelled rows, so its report and "
            f"metric plots are empty. Check that the QC flag values in the "
            f"input match 'pos_flag_values' / 'neg_flag_values'.",
            UserWarning,
            stacklevel=3,
        )
        return True

    warnings.warn(
        f"Evaluation for {where} used labels of a single class "
        f"({labels.len()} rows, all label={present[0]}), so the scores are "
        f"degenerate: precision, recall, f1-score and accuracy are 1.0 by "
        f"construction, and no ROC or precision-recall curve can be computed. "
        f"This usually means every selected row carried the same QC flag; "
        f"check 'pos_flag_values' / 'neg_flag_values' against the flag values "
        f"actually present in the input.",
        UserWarning,
        stacklevel=3,
    )
    return True
