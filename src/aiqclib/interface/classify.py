"""
Main orchestration module for the data classification pipeline.

This module provides the main entry point for executing a comprehensive data classification pipeline.
It orchestrates a series of sequential steps, from initial data loading and preparation to feature
extraction, model prediction, and final result merging. Each step is configured and executed
based on the parameters defined in a provided configuration object.
"""

from aiqclib.common.base.config_base import ConfigBase

from aiqclib.common.loader.classify_loader import (
    load_classify_step1_input_dataset,
    load_classify_step2_summary_dataset,
    load_classify_step3_select_dataset,
    load_classify_step4_locate_dataset,
    load_classify_step5_extract_dataset,
    load_classify_step6_classify_dataset,
    load_classify_step7_concat_dataset,
)


def classify_dataset(config: ConfigBase) -> None:
    """
    Execute a series of steps to classify all observations in the given data set, as defined
    by the provided configuration object.

    This function performs the following steps in sequence:
      1. Load and read the initial input data.
      2. Calculate and write summary statistics.
      3. Label and write selected profiles.
      4. Locate and write target rows.
      5. Extract and write target features.
      6. Use the model to predict labels in the input data.
      7. Merge the results with the original input data.

    :param config: A configuration object specifying the classes and parameters
                   for each step in the dataset preparation and classification process.
    :type config: ConfigBase
    :return: None. The function performs I/O operations and modifies datasets based
             on the configuration but does not return a value.
    :rtype: None
    """
    ds_input = load_classify_step1_input_dataset(config)
    ds_input.read_input_data()

    ds_summary = load_classify_step2_summary_dataset(config, ds_input.input_data)
    ds_summary.calculate_stats()
    ds_summary.write_summary_stats()

    ds_select = load_classify_step3_select_dataset(config, ds_input.input_data)
    ds_select.label_profiles()
    ds_select.write_selected_profiles()

    ds_locate = load_classify_step4_locate_dataset(
        config, ds_input.input_data, ds_select.selected_profiles
    )
    ds_locate.process_targets()
    ds_locate.write_selected_rows()

    ds_extract = load_classify_step5_extract_dataset(
        config,
        ds_input.input_data,
        ds_select.selected_profiles,
        ds_locate.selected_rows,
        ds_summary.summary_stats,
    )
    ds_extract.process_targets()
    ds_extract.write_target_features()

    ds_classify = load_classify_step6_classify_dataset(
        config, ds_extract.target_features
    )
    ds_classify.read_models()
    ds_classify.test_targets()
    ds_classify.write_predictions()
    ds_classify.write_reports()
    ds_classify.write_contingency_tables()
    if ds_classify.base_model.enable_shap:
        ds_classify.write_shap_values()
    ds_classify.create_metric_plots()

    ds_concat = load_classify_step7_concat_dataset(
        config, ds_input.input_data, ds_classify.predictions
    )
    ds_concat.merge_predictions()
    ds_concat.write_merged_predictions()
