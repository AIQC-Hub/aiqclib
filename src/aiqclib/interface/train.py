"""
Orchestration module for the model training and evaluation pipeline.

This module defines the primary workflow for loading datasets, performing model
validation, and executing the final model construction and testing phases based
on a centralized configuration.
"""

from aiqclib.common.base.config_base import ConfigBase
from aiqclib.common.loader.training_loader import (
    load_step1_input_training_set,
    load_step2_model_validation_class,
    load_step4_build_model_class,
)


def train_and_evaluate(config: ConfigBase) -> None:
    """
    Perform a training and evaluation process based on the specified configuration.

    This function orchestrates the end-to-end workflow, including data loading,
    model validation, and final model building and testing.

    Steps:
      1. Load and process input training data.
      2. Validate the model using the specified validation technique (e.g., k-fold).
      3. Build and test the final model, saving results and trained model artifacts.

    :param config: A training configuration object specifying classes and parameters.
    :type config: ConfigBase
    :return: None. The function performs I/O operations and does not return a value.
    :rtype: None
    """
    # Step 1: Input Loading
    ds_input = load_step1_input_training_set(config)
    ds_input.process_targets()

    # Step 2: Model Validation
    ds_valid = load_step2_model_validation_class(config, ds_input.training_sets)
    ds_valid.process_targets()
    ds_valid.write_reports()
    ds_valid.write_contingency_tables()
    ds_valid.create_metric_plots()

    # Step 4: Build and Test Model
    ds_build = load_step4_build_model_class(
        config, ds_input.training_sets, ds_input.test_sets
    )
    ds_build.build_targets()
    ds_build.test_targets()
    ds_build.write_reports()
    ds_build.write_contingency_tables()
    ds_build.create_metric_plots()
    if ds_build.base_model.enable_shap:
        ds_build.write_shap_values()
    ds_build.build_final_model_targets()
    ds_build.write_models()
