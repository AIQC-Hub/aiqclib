"""
This module provides a registry of model classes that can be used
during training or inference steps. Each key in the dictionary
corresponds to a model name (string), and each value is the class
constructor for that model.
"""

from typing import Dict, Type

from aiqclib.common.base.model_base import ModelBase
from aiqclib.train.models.decision_tree import DecisionTree
from aiqclib.train.models.random_forest import RandomForest
from aiqclib.train.models.xgboost import XGBoost
from aiqclib.train.models.logistic_regression import LogisticRegression
from aiqclib.train.models.linear_discriminant_analysis import LinearDiscriminantAnalysis
from aiqclib.train.models.support_vector_machine import SupportVectorMachine
from aiqclib.train.models.k_nearest_neighbors import KNearestNeighbors
from aiqclib.train.models.gaussian_naive_bayes import GaussianNaiveBayes
from aiqclib.train.models.multilayer_perceptron import MultilayerPerceptron

#: A dictionary mapping model names to their corresponding Python classes.
#:
#: The keys are strings (e.g., "XGBoost"), and the values are class objects
#: that inherit from :class:`aiqclib.common.base.model_base.ModelBase`.
#:
#: :type: Dict[str, Type[ModelBase]]
SINGLE_MODEL_REGISTRY: Dict[str, Type[ModelBase]] = {
    "DecisionTree": DecisionTree,
    "DT": DecisionTree,
    "RandomForest": RandomForest,
    "RF": RandomForest,
    "XGBoost": XGBoost,
    "XGB": XGBoost,
    "LogisticRegression": LogisticRegression,
    "Logit": LogisticRegression,
    "LinearDiscriminantAnalysis": LinearDiscriminantAnalysis,
    "LDA": LinearDiscriminantAnalysis,
    "SupportVectorMachine": SupportVectorMachine,
    "SVM": SupportVectorMachine,
    "KNearestNeighbors": KNearestNeighbors,
    "KNN": KNearestNeighbors,
    "GaussianNaiveBayes": GaussianNaiveBayes,
    "GNB": GaussianNaiveBayes,
    "MultilayerPerceptron": MultilayerPerceptron,
    "MLP": MultilayerPerceptron,
}
