"""
This module defines the `SplitDataSetAll` class, which is responsible for partitioning
feature data into training and test sets. It provides functionality for random sampling,
k-fold cross-validation index assignment, and column cleanup for Copernicus CTD datasets.
"""

from typing import Optional, Dict

import numpy as np
import polars as pl

from aiqclib.common.base.config_base import ConfigBase
from aiqclib.prepare.step6_split_dataset.split_base import SplitDataSetBase


class SplitDataSetAll(SplitDataSetBase):
    """
    A subclass of :class:`SplitDataSetBase` that splits feature data into
    training and test sets for Copernicus CTD data.

    This class performs the following tasks:
      - Randomly samples a fraction of rows for the test set.
      - Ensures matching positive and negative rows are grouped by shared
        identifiers (e.g., ``pair_id``).
      - Splits out the remainder into a training set.
      - Assigns k-fold indices to the training set rows.
      - Optionally drops columns that are not required for subsequent analysis.

    .. note::
       This class, :class:`SplitDataSetAll`, is specifically designed to split
       feature data into training and test sets with particular handling for
       Copernicus CTD data.
    """

    expected_class_name: str = "SplitDataSetAll"

    def __init__(
        self,
        config: ConfigBase,
        target_features: Optional[Dict[str, pl.DataFrame]] = None,
    ) -> None:
        """
        Initialize the dataset splitting class with configuration
        and target features.

        :param config: A dataset configuration object that specifies
                       paths, test-set fraction, and k-fold details.
        :type config: :class:`aiqclib.common.base.config_base.ConfigBase`
        :param target_features: A dictionary mapping target names to Polars
                                DataFrames containing extracted features.
                                Defaults to None.
        :type target_features: Optional[Dict[str, pl.DataFrame]]
        """
        super().__init__(config=config, target_features=target_features)

        #: Column names used for intermediate processing (e.g., to maintain
        #: matching references between positive and negative rows).
        self.drop_col_names = [
            "profile_id",
            "pair_id",
        ]

    def split_test_set(self, target_name: str) -> None:
        """
        Split the specified target's DataFrame into training and test sets.

        1. A random fraction of rows labeled 1 (positive) is sampled to form
           the test set.
        2. Rows labeled 0 (negative) with matching ``pair_id`` are joined
           to that test set.
        3. The remaining rows form the training set.

        :param target_name: The target name identifying which DataFrame in
                            :attr:`target_features` to split.
        :type target_name: str
        """
        test_set_fraction = self.get_test_set_fraction()

        pos_test_set = (
            self.target_features[target_name]
            .filter(pl.col("label") == 1)
            .sample(fraction=test_set_fraction, shuffle=True)
        )

        neg_test_set = (
            self.target_features[target_name]
            .filter(pl.col("label") == 0)
            .sample(fraction=test_set_fraction, shuffle=True)
        )

        test_set = pos_test_set.vstack(neg_test_set)
        self.test_sets[target_name] = test_set.select(
            ["row_id", pl.all().exclude("row_id")]
        )

        pos_training_set = (
            self.target_features[target_name].filter(pl.col("label") == 1)
        ).join(pos_test_set, on="row_id", how="anti")

        neg_training_set = (
            self.target_features[target_name].filter(pl.col("label") == 0)
        ).join(neg_test_set, on="row_id", how="anti")

        training_set = pos_training_set.vstack(neg_training_set)
        self.training_sets[target_name] = training_set.select(
            ["row_id", pl.all().exclude("row_id")]
        )

    def add_k_fold(self, target_name: str) -> None:
        """
        Assign a k-fold identifier to each row in the training set for cross-validation.

        1. Extracts rows labeled 1 (positive) and unevenly distributes them across
           the specified number of folds.
        2. Joins negative rows based on ``pair_id`` so they share the same fold
           assignment.

        :param target_name: The target name identifying the training set
                            within :attr:`training_sets`.
        :type target_name: str
        """
        k_fold = self.get_k_fold()
        pos_training_set = self.training_sets[target_name].filter(pl.col("label") == 1)
        neg_training_set = self.training_sets[target_name].filter(pl.col("label") == 0)
        df_pos_size = pos_training_set.shape[0]
        df_neg_size = neg_training_set.shape[0]

        pos_n_per_value = df_pos_size // k_fold
        neg_n_per_value = df_neg_size // k_fold
        pos_k_values = np.array(
            [i for i in range(1, k_fold + 1) for _ in range(pos_n_per_value)]
        )
        neg_k_values = np.array(
            [i for i in range(1, k_fold + 1) for _ in range(neg_n_per_value)]
        )

        pos_remaining = df_pos_size % k_fold
        neg_remaining = df_neg_size % k_fold
        pos_k_values = np.concatenate(
            [pos_k_values, np.random.choice(range(1, k_fold + 1), pos_remaining)]
        )
        neg_k_values = np.concatenate(
            [neg_k_values, np.random.choice(range(1, k_fold + 1), neg_remaining)]
        )

        np.random.shuffle(pos_k_values)
        np.random.shuffle(neg_k_values)
        pos_training_set = pos_training_set.with_columns(
            pl.Series("k_fold", pos_k_values)
        )
        neg_training_set = neg_training_set.with_columns(
            pl.Series("k_fold", neg_k_values)
        )

        training_set = pos_training_set.vstack(neg_training_set)

        # The correct way to reorder columns is to explicitly select them.
        cols_to_front = [
            "k_fold",
            "row_id",
            "platform_code",
            "profile_no",
            "observation_no",
        ]
        self.training_sets[target_name] = training_set.select(
            cols_to_front + [pl.all().exclude(cols_to_front)]
        )

    def drop_columns(self, target_name: str) -> None:
        """
        Remove specified working columns from both the training and test sets,
        leaving only the essential columns for subsequent steps.

        :param target_name: The target name identifying which training and test sets
                            to modify.
        :type target_name: str
        """
        self.training_sets[target_name] = self.training_sets[target_name].drop(
            self.drop_col_names
        )
        self.test_sets[target_name] = self.test_sets[target_name].drop(
            self.drop_col_names
        )
