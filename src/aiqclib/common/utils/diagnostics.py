"""
Warnings for results that are computed successfully but carry no information.

Some inputs produce metrics that look perfect while measuring nothing. The
clearest case is an evaluation set whose labels are all the same class: every
prediction of that class is correct, so precision, recall, f1 and accuracy all
come out as 1.0, and no ROC or precision-recall curve can be drawn at all. The
numbers are not wrong, they are meaningless, and nothing in the output says so.

This module raises that kind of situation as a warning at the point it is
detected, naming the target and what to check, so it is visible while the run is
happening rather than inferred later from a suspiciously perfect report.
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
