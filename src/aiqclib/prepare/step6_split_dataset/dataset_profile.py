"""
This module defines the SplitDataSetProfile class, the profile-level variant
of :class:`SplitDataSetAll`. Binary profile labels reuse the stratified
splitting of the parent class; proportion labels (floats in [0, 1]) get a
plain random test split and uniform k-fold assignment, since class-based
stratification does not apply to continuous labels.
"""

import numpy as np
import polars as pl

from aiqclib.common.constants import ID_COLUMNS, existing_columns
from aiqclib.prepare.step6_split_dataset.dataset_all import SplitDataSetAll


class SplitDataSetProfile(SplitDataSetAll):
    """
    A subclass of :class:`SplitDataSetAll` that splits profile-level feature
    data into training and test sets.

    When the ``label`` column is a float (``label_mode: proportion``), the
    test set is a plain random sample and k-fold indices are assigned
    uniformly at random. Integer (binary) labels fall through to the
    label-stratified logic inherited from :class:`SplitDataSetAll`.
    """

    expected_class_name: str = "SplitDataSetProfile"

    def _has_proportion_label(self, df: pl.DataFrame) -> bool:
        """
        Return True when the frame's ``label`` column holds proportions.

        :param df: The frame whose label dtype to inspect.
        :type df: pl.DataFrame
        :return: True for a float label column (proportion mode).
        :rtype: bool
        """
        return df.schema["label"].is_float()

    def split_test_set(self, target_name: str) -> None:
        """
        Split the target's DataFrame into training and test sets.

        Proportion labels: a random fraction of all rows forms the test set
        and the remainder the training set. Binary labels: inherited
        per-class sampling.

        :param target_name: The target name identifying which DataFrame in
                            :attr:`target_features` to split.
        :type target_name: str
        """
        features = self.target_features[target_name]
        if not self._has_proportion_label(features):
            super().split_test_set(target_name)
            return

        test_set = features.sample(fraction=self.get_test_set_fraction(), shuffle=True)
        training_set = features.join(test_set, on="row_id", how="anti")

        self.test_sets[target_name] = test_set.select(
            ["row_id", pl.all().exclude("row_id")]
        )
        self.training_sets[target_name] = training_set.select(
            ["row_id", pl.all().exclude("row_id")]
        )

    def add_k_fold(self, target_name: str) -> None:
        """
        Assign a k-fold identifier to each training-set row.

        Proportion labels: folds are assigned uniformly at random (balanced
        fold sizes, shuffled). Binary labels: inherited stratified
        assignment.

        :param target_name: The target name identifying the training set
                            within :attr:`training_sets`.
        :type target_name: str
        """
        training_set = self.training_sets[target_name]
        if not self._has_proportion_label(training_set):
            super().add_k_fold(target_name)
            return

        k_fold = self.get_k_fold()
        k_values = (np.arange(training_set.shape[0]) % k_fold + 1).astype(np.int64)
        np.random.shuffle(k_values)
        training_set = training_set.with_columns(pl.Series("k_fold", k_values))

        cols_to_front = existing_columns(training_set, ["k_fold"] + ID_COLUMNS)
        self.training_sets[target_name] = training_set.select(
            cols_to_front + [pl.all().exclude(cols_to_front)]
        )
