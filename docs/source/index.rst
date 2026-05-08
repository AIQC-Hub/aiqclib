Welcome to the `aiqclib` documentation!
=======================================

``aiqclib`` is a Python library that provides a configuration-driven workflow for machine learning, simplifying dataset preparation, model training, and data classification. It is a core component of the AIQC project.

.. note::
   To unlock the full potential of your model building processes, comprehensive configuration is key. For a streamlined initial test or to explore the library with base settings, see the :doc:`how-to/quick_start` guide.

Workflow of `aiqclib`
---------------------------------------------

The library is designed around a three-stage workflow:

1.  **Dataset Preparation:** Prepare feature datasets from raw data and generate training, validation, and test data sets.
2.  **Training & Evaluation:** Train machine learning models and evaluate their performance using cross-validation.
3.  **Classification:** Apply a trained model to classify new, unseen data.

Each stage is controlled by a YAML configuration file, allowing you to define and reproduce your entire workflow with ease.

Algorithms supported by `aiqclib`
---------------------------------------------

================================ ================================= ============ ======================
Category                         Algorithm                         Short Name   Method
================================ ================================= ============ ======================
Tree-Based & Ensemble            **XGBoost**                       XGB          Ensemble (Boosting)
\                                **Random Forest**                 RF           Ensemble (Bagging)
\                                **Decision Tree**                 DT           Tree
Linear & Geometric               **Logistic Regression**           Logit        Linear
\                                **Linear Discriminant Analysis**  LDA          Linear / Statistical
\                                **Support Vector Machine**        SVM          Geometric
Instance-Based                   **K-Nearest Neighbors**           KNN          Distance-based
Probabilistic                    **Gaussian Naive Bayes**          GNB          Probabilistic
Neural Network                   **Multilayer Perceptron**         MLP          Neural Network
================================ ================================= ============ ======================

Project links
---------------------------------------------

  - `GitHub Repository <https://github.com/AIQC-Hub/aiqclib>`_
  - `Issue Tracker <https://github.com/AIQC-Hub/aiqclib/issues>`_
  - `Pull Requests <https://github.com/AIQC-Hub/aiqclib/pulls>`_

----------

These tutorials provide a step-by-step guide to the core workflows of the library. If you are new to ``aiqclib``, start here.

.. toctree::
   :maxdepth: 2
   :caption: 📘 Getting Started

   tutorial/overview
   tutorial/installation
   tutorial/preparation
   tutorial/training
   tutorial/classification

----------

This section provides practical examples and solutions for common tasks related to using ``aiqclib``.

.. toctree::
   :maxdepth: 2
   :caption: 💡 How-To Guides

   how-to/quick_start
   how-to/data_preprocessing_utilities
   how-to/algorithm_selection
   how-to/shap_values
   how-to/down_sampling_negative
   how-to/feature_normalization
   how-to/selecting_specific_configurations

----------

This section provides detailed reference information for all parameters in the YAML configuration files.

.. toctree::
   :maxdepth: 2
   :caption: ⚙️ Configuration

   configuration/preparation
   configuration/training
   configuration/classification

----------

Understanding the input features is crucial for building effective models. This section provides a detailed explanation of each feature used for model training.

.. toctree::
   :maxdepth: 2
   :caption: 📊 Features

   features/location
   features/day_of_year
   features/basic_values
   features/profile_summary_stats
   features/neigbouring_values

----------

For in-depth information on specific functions, classes, and methods, consult the API documentation.

.. toctree::
   :maxdepth: 4
   :caption: 🧩 API Reference

   api/modules