"""
This module provides the KFoldValidation class, an implementation of k-fold cross-validation
for model training and evaluation. It extends ValidationBase to perform iterative model
building and testing across defined data folds, accumulating performance reports.
"""

from typing import Optional, List, Dict
import copy

import polars as pl

from aiqclib.common.base.config_base import ConfigBase
from aiqclib.train.step2_validate_model.validate_base import ValidationBase


class KFoldValidation(ValidationBase):
    """
    A subclass of :class:`ValidationBase` that performs k-fold cross-validation
    on training sets.

    This class iterates over the specified number of folds, trains
    (builds) the model on all folds except one, then tests it on the
    held-out fold. Results are accumulated in :attr:`reports`.
    """

    expected_class_name: str = "KFoldValidation"

    def __init__(
        self,
        config: ConfigBase,
        training_sets: Optional[Dict[str, pl.DataFrame]] = None,
    ) -> None:
        """
        Initialize the k-fold validation process.

        :param config: A training configuration object containing
                       model parameters, file paths, and other
                       validation settings.
        :type config: ConfigBase
        :param training_sets: A dictionary where keys are target names and values are
                              Polars DataFrames of labeled data. Each DataFrame must
                              contain a column named ``k_fold`` indicating the fold
                              assignment for each row. Defaults to None.
        :type training_sets: Optional[Dict[str, pl.DataFrame]]
        """
        super().__init__(config=config, training_sets=training_sets)

        #: The default number of folds if none is specified in the config.
        self.default_k_fold: int = 10
        self.drop_cols = [
            "k_fold",
            "row_id",
            "platform_code",
            "profile_no",
            "observation_no",
        ]

    def get_k_fold(self) -> int:
        """
        Retrieve the number of folds to use for cross-validation from
        the ``validate`` section of the YAML config, or fall back
        to :attr:`default_k_fold`.

        :return: The number of folds for k-fold cross-validation.
        :rtype: int
        """
        return (
            self.config.get_step_params("validate").get("k_fold", self.default_k_fold)
            or self.default_k_fold
        )

    def validate(self, target_name: str) -> None:
        """
        Conduct k-fold cross-validation for the given target name,
        storing model objects, test results, and contingency tables in
        :attr:`models`, :attr:`reports`, and :attr:`contingency_tables`.

        For each fold out of :meth:`get_k_fold`:

          1. Reload or re-initialize the model using :meth:`load_base_model`.
          2. Set ``base_model.k`` to the fold index.
          3. Build the model using all training data except rows in the current fold.
          4. Test the model on the held-out fold.
          5. Accumulate test results and contingency tables.

        :param target_name: The identifier for which target dataset to validate,
                            referring to the corresponding DataFrame within
                            :attr:`training_sets`.
        :type target_name: str
        """
        self.models[target_name] = []
        reports: List[pl.DataFrame] = []
        contingency_tables: List[pl.DataFrame] = []

        k_fold: int = self.get_k_fold()
        for k in range(k_fold):
            self.load_base_model()
            current_fold_model = copy.deepcopy(self.base_model)

            current_fold_model.k = k + 1
            current_fold_model.training_set = (
                self.training_sets[target_name]
                .filter(pl.col("k_fold") != (k + 1))
                .drop(self.drop_cols)
            )
            current_fold_model.build()
            self.models[target_name].append(current_fold_model)

            current_fold_model.test_set = (
                self.training_sets[target_name]
                .filter(pl.col("k_fold") == (k + 1))
                .drop(self.drop_cols)
            )
            current_fold_model.test()
            reports.append(current_fold_model.report)

            if current_fold_model.contingency_table is not None:
                contingency_tables.append(current_fold_model.contingency_table)

        self.reports[target_name] = pl.concat(reports)

        if contingency_tables:
            self.contingency_tables[target_name] = pl.concat(contingency_tables)
