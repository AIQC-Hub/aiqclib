"""Shared parametrize table for the 9 supported model wrappers.

This module exists so the per-model fan-out lives in exactly one place. Test
files that previously contained nine near-identical ``TestXxx`` classes (one
per model) instead import ``MODEL_CASES`` and use
``@pytest.mark.parametrize("case", MODEL_CASES, ids=lambda c: c.config_name)``
on a single class with the four (or so) shared test methods.

Different test files need different subsets of the fields:
- ``test_training_step2_validate_a.py`` and ``test_training_step4_build_a.py``
  only need ``wrapper_cls`` and ``config_name`` to check ``isinstance`` against
  ``ds.base_model``.
- ``test_training_step4_build_a.py``'s ``test_write_model`` and
  ``test_classify_step6_classify_all.py``'s per-model tests need
  ``joblib_suffix`` to either produce per-model output filenames (step4) or
  match production fixture names (step6's ``model_{tgt}_{suffix}.joblib``).
- ``test_training_models.py`` exercises the wrapper APIs directly and needs
  the full set (``sklearn_cls``, ``defaults``, ``override``, ``missing``).

Rather than maintain multiple tables, ``ModelCase`` carries every field as
required (``config_name``, ``wrapper_cls``) or optional (everything else).
Tests pick the attributes they need.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Type

import xgboost as xgb
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as SklearnLDA
from sklearn.ensemble import RandomForestClassifier as SklearnRF
from sklearn.linear_model import LogisticRegression as SklearnLR
from sklearn.naive_bayes import GaussianNB as SklearnGNB
from sklearn.neighbors import KNeighborsClassifier as SklearnKNN
from sklearn.neural_network import MLPClassifier as SklearnMLP
from sklearn.svm import SVC as SklearnSVC
from sklearn.tree import DecisionTreeClassifier as SklearnDT

from aiqclib.train.models.decision_tree import DecisionTree
from aiqclib.train.models.gaussian_naive_bayes import GaussianNaiveBayes
from aiqclib.train.models.k_nearest_neighbors import KNearestNeighbors
from aiqclib.train.models.linear_discriminant_analysis import LinearDiscriminantAnalysis
from aiqclib.train.models.logistic_regression import LogisticRegression
from aiqclib.train.models.multilayer_perceptron import MultilayerPerceptron
from aiqclib.train.models.random_forest import RandomForest
from aiqclib.train.models.support_vector_machine import SupportVectorMachine
from aiqclib.train.models.xgboost import XGBoost


@dataclass(frozen=True)
class ModelCase:
    """One row of the model-wrapper test table.

    Required fields are enough for ``isinstance`` checks and config wiring.
    Optional fields are only consumed by tests that need them.

    Attributes:
        config_name: String used in YAML's ``step_class_set.steps.model``.
        wrapper_cls: The aiqclib wrapper class.
        joblib_suffix: Short string used in per-model output filenames in
            step4 (``test_model_{tgt}_{suffix}.joblib``) and to read existing
            per-model fixture files in step6 (``model_{tgt}_{suffix}.joblib``).
            Matches the production fixture naming convention.
        sklearn_cls: Underlying scikit/xgboost class ``_get_model_class`` returns.
        defaults: Subset of default params that must be present with these values.
        override: Params injected via config and verified to round-trip.
        missing: Params that must NOT appear in defaults (e.g. ``n_jobs`` for LDA,
            ``penalty`` for LogisticRegression after the sklearn 1.8 deprecation).
    """

    config_name: str
    wrapper_cls: Type
    joblib_suffix: Optional[str] = None
    sklearn_cls: Optional[Type] = None
    defaults: Optional[Dict[str, Any]] = None
    override: Optional[Dict[str, Any]] = None
    missing: Tuple[str, ...] = field(default_factory=tuple)


# The canonical list. Order matters only for human readability; pytest uses
# config_name as the test id.
MODEL_CASES: list[ModelCase] = [
    ModelCase(
        config_name="XGBoost",
        wrapper_cls=XGBoost,
        joblib_suffix="xgb",
        sklearn_cls=xgb.XGBClassifier,
        defaults={"n_estimators": 100, "n_jobs": -1},
        override={"max_depth": 10, "n_jobs": 4},
    ),
    ModelCase(
        config_name="LogisticRegression",
        wrapper_cls=LogisticRegression,
        joblib_suffix="logit",
        sklearn_cls=SklearnLR,
        defaults={"C": 1.0, "solver": "lbfgs", "l1_ratio": 0},
        override={"C": 0.5, "max_iter": 500},
        missing=("penalty",),
    ),
    ModelCase(
        config_name="LinearDiscriminantAnalysis",
        wrapper_cls=LinearDiscriminantAnalysis,
        joblib_suffix="lda",
        sklearn_cls=SklearnLDA,
        defaults={"solver": "svd"},
        override={"tol": 1.0e-5},
        missing=("n_jobs",),
    ),
    ModelCase(
        config_name="SVM",
        wrapper_cls=SupportVectorMachine,
        joblib_suffix="svm",
        sklearn_cls=SklearnSVC,
        defaults={"kernel": "linear", "probability": True},
        override={"C": 0.5, "kernel": "rbf"},
    ),
    ModelCase(
        config_name="DecisionTree",
        wrapper_cls=DecisionTree,
        joblib_suffix="dt",
        sklearn_cls=SklearnDT,
        defaults={"criterion": "gini", "splitter": "best"},
        override={"max_depth": 5, "min_samples_split": 4},
    ),
    ModelCase(
        config_name="RandomForest",
        wrapper_cls=RandomForest,
        joblib_suffix="rf",
        sklearn_cls=SklearnRF,
        defaults={"n_estimators": 100, "n_jobs": -1},
        override={"n_estimators": 50, "max_features": "log2"},
    ),
    ModelCase(
        config_name="KNearestNeighbors",
        wrapper_cls=KNearestNeighbors,
        joblib_suffix="knn",
        sklearn_cls=SklearnKNN,
        defaults={"n_neighbors": 5, "n_jobs": -1},
        override={"n_neighbors": 10, "weights": "distance"},
    ),
    ModelCase(
        config_name="GaussianNaiveBayes",
        wrapper_cls=GaussianNaiveBayes,
        joblib_suffix="gnb",
        sklearn_cls=SklearnGNB,
        defaults={"var_smoothing": 1e-9},
        override={"var_smoothing": 1e-5},
    ),
    ModelCase(
        config_name="MLP",
        wrapper_cls=MultilayerPerceptron,
        joblib_suffix="mlp",
        sklearn_cls=SklearnMLP,
        defaults={"hidden_layer_sizes": (50,), "activation": "relu"},
        override={"hidden_layer_sizes": (50, 50), "learning_rate_init": 0.01},
    ),
]
