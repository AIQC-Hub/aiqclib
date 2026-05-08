Overview
========

Welcome to the ``aiqclib`` tutorial! This library provides a robust and streamlined framework for building and deploying machine learning models, specifically designed for applications that involve quality control of delayed-mode data and similar predictive tasks. It simplifies complex machine learning operations (MLOps) workflows by breaking them down into a clear, configurable three-stage process.

Basic Usage: The Three-Stage Workflow
-------------------------------------

``aiqclib`` leverages powerful YAML configuration files to define every aspect of your machine learning pipeline, enabling reproducibility and easy experimentation. Once these configuration files are set up, executing a complete end-to-end workflow is straightforward:

1.  **Dataset Preparation:**
    This initial stage transforms your raw input data into a clean, feature-engineered, and properly split dataset (training, validation, test sets) ready for model building.

    .. code-block:: python

       import aiqclib as aq

       prepare_config = aq.read_config("/path/to/prepare_config.yaml")
       aq.create_training_dataset(prepare_config)

2.  **Training & Evaluation:**
    In this stage, a machine learning model is trained and rigorously evaluated using the prepared dataset. This typically includes cross-validation and hyperparameter tuning to find the best performing model.

    .. code-block:: python

       training_config = aq.read_config("/path/to/training_config.yaml")
       aq.train_and_evaluate(training_config)

3.  **Classification (Inference):**
    The final stage applies your trained model to new, unseen data to generate predictions or classifications. This is where your model goes from development to practical application.

    .. code-block:: python

       classification_config = aq.read_config("/path/to/classification_config.yaml")
       aq.classify_dataset(classification_config)

Objectives
-----------------------------

You will learn how to run all three stages of ``aiqclib`` by creating stage-specific configuration files. This tutorial lets you create three classifiers for ``temp`` (temperature), ``psal`` (salinity), and ``pres`` (pressure) to predict QC labels for the corresponding variables.

Next Steps
----------

Ready to get started? The next pages will guide you through the initial setup and then dive into the details of configuring each stage of your machine learning workflow with ``aiqclib``.

Proceed to the next tutorial: :doc:`./installation`.
