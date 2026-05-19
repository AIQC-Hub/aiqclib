"""Unit tests for ``load_model_class`` enumerated across all 9 single-model wrappers.

This file is the sibling of ``test_common_loaders_model.py``, which focuses
on XGBoost + ModelSuite. This file covers the other 8 single-model wrappers
(LogisticRegression, LinearDiscriminantAnalysis, SupportVectorMachine,
DecisionTree, RandomForest, KNearestNeighbors, GaussianNaiveBayes,
MultilayerPerceptron) — plus XGBoost again, so that all 9 wrappers can be
verified from a single source of truth.

Coverage:
- Each of the 9 wrappers loads correctly via both its long-form name
  (``XGBoost``, ``LogisticRegression``, ...) and its short alias
  (``XGB``, ``Logit``, ...) — 18 (config_name, wrapper_class) cases total
- An invalid config string raises ValueError
- Calling build/predict/create_report on a freshly-loaded wrapper (no
  training_set, no test_set, no predictions) raises ValueError

Refactored from a ``unittest.TestCase`` class with 9 nearly-identical test
methods (one per wrapper), each containing two ``self.config.data[...]`` +
``assertIsInstance`` blocks. The 9 methods collapse to 1 parametrized test
over 18 cases.

The 5 error tests at the bottom are duplicated in
``test_common_loaders_model.py`` — both files call ``load_model_class``
with the default config (which yields XGBoost) and exercise the same error
paths. Preserving both for maximum regression coverage.
"""

import pytest

from aiqclib.common.loader.model_loader import load_model_class
from aiqclib.train.models.decision_tree import DecisionTree
from aiqclib.train.models.gaussian_naive_bayes import GaussianNaiveBayes
from aiqclib.train.models.k_nearest_neighbors import KNearestNeighbors
from aiqclib.train.models.linear_discriminant_analysis import LinearDiscriminantAnalysis
from aiqclib.train.models.logistic_regression import LogisticRegression
from aiqclib.train.models.multilayer_perceptron import MultilayerPerceptron
from aiqclib.train.models.random_forest import RandomForest
from aiqclib.train.models.support_vector_machine import SupportVectorMachine
from aiqclib.train.models.xgboost import XGBoost


# ---------------------------------------------------------------------------
# Parametrize cases: all 9 single-model wrappers × 2 alias forms = 18 entries.
#
# Could be derived from MODEL_CASES in _model_cases.py, but MODEL_CASES only
# tracks the long-form name. Adding an ``aliases`` field there would let
# multiple files share this mapping — flagged as a possible future refactor.
# For now, the list is local to this file.
# ---------------------------------------------------------------------------

_MODEL_NAMES = [
    ("XGBoost",                    XGBoost),
    ("XGB",                        XGBoost),
    ("LogisticRegression",         LogisticRegression),
    ("Logit",                      LogisticRegression),
    ("LinearDiscriminantAnalysis", LinearDiscriminantAnalysis),
    ("LDA",                        LinearDiscriminantAnalysis),
    ("SupportVectorMachine",       SupportVectorMachine),
    ("SVM",                        SupportVectorMachine),
    ("DecisionTree",               DecisionTree),
    ("DT",                         DecisionTree),
    ("RandomForest",               RandomForest),
    ("RF",                         RandomForest),
    ("KNearestNeighbors",          KNearestNeighbors),
    ("KNN",                        KNearestNeighbors),
    ("GaussianNaiveBayes",         GaussianNaiveBayes),
    ("GNB",                        GaussianNaiveBayes),
    ("MultilayerPerceptron",       MultilayerPerceptron),
    ("MLP",                        MultilayerPerceptron),
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestModelClassLoader:
    """Tests for load_model_class across all 9 single-model wrappers."""

    @pytest.mark.parametrize(
        "config_name, expected_class",
        _MODEL_NAMES,
        ids=[case[0] for case in _MODEL_NAMES],
    )
    def test_load_model_class_by_name(
        self, config_name, expected_class, training_config_001,
    ):
        """Each (config_name, expected_class) pair produces the right wrapper.

        Verifies that load_model_class correctly maps both the long and short
        forms to the same underlying wrapper class, for all 9 single-model
        wrappers. ``pytest -k XGB`` and ``pytest -k LogisticRegression``
        style filters still work via the parametrize IDs.
        """
        training_config_001.data["step_class_set"]["steps"]["model"] = config_name
        ds = load_model_class(training_config_001)
        assert isinstance(ds, expected_class)

    def test_load_model_invalid_config(self, training_config_001):
        """An unrecognized model name in the config raises ValueError."""
        training_config_001.data["step_class_set"]["steps"]["model"] = "invalid_model_name"
        with pytest.raises(ValueError):
            _ = load_model_class(training_config_001)

    # ----- Error states on the loaded wrapper (default XGBoost) -----

    def test_build_model_empty_training_set(self, training_config_001):
        """build() with no training_set raises ValueError."""
        ds = load_model_class(training_config_001)
        with pytest.raises(ValueError):
            ds.build()

    def test_predict_model_empty_test_set(self, training_config_001):
        """predict() with no test_set raises ValueError."""
        ds = load_model_class(training_config_001)
        with pytest.raises(ValueError):
            ds.predict()

    def test_create_report_empty_test_set(self, training_config_001):
        """create_report() with no test_set raises ValueError."""
        ds = load_model_class(training_config_001)
        with pytest.raises(ValueError):
            ds.create_report()

    def test_create_report_empty_predictions(self, training_config_001):
        """create_report() with test_set set but no predictions raises ValueError.

        Sets test_set to ``{}`` to bypass the "test_set not set" check and
        force the path that validates ``predictions`` instead.
        """
        ds = load_model_class(training_config_001)
        ds.test_set = {}
        with pytest.raises(ValueError):
            ds.create_report()