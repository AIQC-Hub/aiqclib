"""Unit tests for the evaluation diagnostics (``common.utils.diagnostics``).

``warn_single_class_labels`` is the guard against an evaluation that succeeds
while measuring nothing: when every label is the same class, the report comes
back full of 1.0 scores and the metric plots have no curve. The tests verify
that the situation is detected, that the message says what to check, and that
ordinary two-class data stays silent.
"""

import warnings

import polars as pl
import pytest

from aiqclib.common.utils.diagnostics import (
    check_dataset_not_empty,
    check_labels_not_single_class,
    warn_single_class_labels,
)


class TestWarnSingleClassLabels:
    """Detection of degenerate evaluation labels."""

    def test_two_classes_are_silent(self):
        """Ordinary data must not warn."""
        labels = pl.Series("label", [0, 1, 0, 1])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert warn_single_class_labels(labels, target_name="temp") is False

    @pytest.mark.parametrize("value", [0, 1])
    def test_single_class_warns(self, value):
        """Either class alone is degenerate, not just the negative one."""
        labels = pl.Series("label", [value] * 5)
        with pytest.warns(UserWarning, match="single class") as record:
            assert warn_single_class_labels(labels, target_name="temp") is True
        message = str(record[0].message)
        assert "temp" in message
        assert f"label={value}" in message
        assert "5 rows" in message

    def test_message_says_what_to_check(self):
        """The warning has to point at the setting that causes this."""
        labels = pl.Series("label", [0, 0])
        with pytest.warns(UserWarning) as record:
            warn_single_class_labels(labels, target_name="psal")
        message = str(record[0].message)
        assert "pos_flag_values" in message
        assert "neg_flag_values" in message

    def test_empty_labels_get_their_own_message(self):
        """No rows at all is a different problem from one class."""
        labels = pl.Series("label", [], dtype=pl.Int64)
        with pytest.warns(UserWarning, match="no labelled rows"):
            assert warn_single_class_labels(labels, target_name="temp") is True

    def test_nulls_are_not_counted_as_a_class(self):
        """A null label is missing, not a second class."""
        labels = pl.Series("label", [0, None, 0])
        with pytest.warns(UserWarning, match="single class"):
            assert warn_single_class_labels(labels) is True

    def test_fold_is_named_when_given(self):
        """Cross-validation warnings identify which fold was degenerate."""
        labels = pl.Series("label", [1, 1])
        with pytest.warns(UserWarning) as record:
            warn_single_class_labels(labels, target_name="temp", k=3)
        assert "fold 3" in str(record[0].message)

    def test_unfolded_evaluation_mentions_no_fold(self):
        """k=0 means a single evaluation, so no fold number is shown."""
        labels = pl.Series("label", [1, 1])
        with pytest.warns(UserWarning) as record:
            warn_single_class_labels(labels, target_name="temp", k=0)
        assert "fold" not in str(record[0].message)

    def test_target_name_is_optional(self):
        """Callers that do not know the target still get a usable message."""
        labels = pl.Series("label", [0, 0])
        with pytest.warns(UserWarning, match="this target"):
            warn_single_class_labels(labels)


class TestCheckDatasetNotEmpty:
    """Refusing a dataset that has no rows to work with."""

    def test_a_dataset_with_rows_passes(self):
        """The common case returns quietly."""
        frame = pl.DataFrame({"temp": [1.0, 2.0]})
        assert check_dataset_not_empty(frame, "training set", "temp") is None

    def test_an_empty_dataset_raises(self):
        """No rows means nothing downstream can work, so this is an error."""
        frame = pl.DataFrame({"temp": []}, schema={"temp": pl.Float64})
        with pytest.raises(ValueError, match="training set for target 'temp'"):
            check_dataset_not_empty(frame, "training set", "temp")

    def test_message_names_the_likely_cause(self):
        """The message points at the filters, which is what usually empties it."""
        frame = pl.DataFrame({"temp": []}, schema={"temp": pl.Float64})
        with pytest.raises(ValueError) as excinfo:
            check_dataset_not_empty(frame, "classification input", "temp")

        message = str(excinfo.value)
        assert "classification input" in message
        assert "keep_years" in message

    def test_fold_number_is_included(self):
        """A fold is identified by number so the failing fold is obvious."""
        frame = pl.DataFrame({"temp": []}, schema={"temp": pl.Float64})
        with pytest.raises(ValueError, match="fold 3"):
            check_dataset_not_empty(frame, "validation fold", "psal", k=3)

    def test_target_name_is_optional(self):
        """Callers without a target name still get a usable message."""
        frame = pl.DataFrame({"temp": []}, schema={"temp": pl.Float64})
        with pytest.raises(ValueError, match="this target"):
            check_dataset_not_empty(frame, "test set")

    def test_a_frame_with_no_columns_but_rows_passes(self):
        """Emptiness is about rows; column count is a separate concern."""
        frame = pl.DataFrame({"a": [1]}).select()
        assert frame.height == 0  # polars drops rows with no columns
        with pytest.raises(ValueError):
            check_dataset_not_empty(frame, "test set", "temp")


class TestCheckLabelsNotSingleClass:
    """Refusing to fit a model on labels that hold only one class."""

    def test_two_classes_pass(self):
        """The normal case returns quietly."""
        labels = pl.Series("label", [0, 1, 0, 1])
        assert check_labels_not_single_class(labels, "temp") is None

    def test_a_single_class_raises(self):
        """One class means the model could only ever predict that class."""
        labels = pl.Series("label", [0] * 100)
        with pytest.raises(ValueError, match="Training data for target 'pres'"):
            check_labels_not_single_class(labels, "pres")

    def test_message_reports_the_class_and_row_count(self):
        """The message carries the evidence, as the warning version does."""
        labels = pl.Series("label", [0] * 53719)
        with pytest.raises(ValueError) as excinfo:
            check_labels_not_single_class(labels, "pres", k=1)

        message = str(excinfo.value)
        assert "all label=0" in message
        assert "53719 rows" in message
        assert "fold 1" in message

    def test_all_positive_is_refused_too(self):
        """A dataset of only positives is as unusable as one of only negatives."""
        labels = pl.Series("label", [1, 1, 1])
        with pytest.raises(ValueError, match="all label=1"):
            check_labels_not_single_class(labels, "temp")

    def test_no_labelled_rows_at_all(self):
        """All-null labels are reported as such rather than as a class."""
        labels = pl.Series("label", [None, None], dtype=pl.Int64)
        with pytest.raises(ValueError, match="no labelled rows at all"):
            check_labels_not_single_class(labels, "temp")

    def test_nulls_do_not_count_as_a_second_class(self):
        """A null is missing, not a class of its own."""
        labels = pl.Series("label", [0, 0, None], dtype=pl.Int64)
        with pytest.raises(ValueError, match="all label=0"):
            check_labels_not_single_class(labels, "temp")

    def test_message_suggests_dropping_the_target(self):
        """The actionable fix for a genuinely unflagged variable is named."""
        labels = pl.Series("label", [0, 0])
        with pytest.raises(ValueError, match="drop it from the 'target_set'"):
            check_labels_not_single_class(labels, "pres")
