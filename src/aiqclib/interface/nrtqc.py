"""
Main orchestration module for the Near-Real Time Quality Control (NRT QC)
pipeline.

This module provides the main entry point for executing the NRT QC pipeline:
reading the input data, applying the configured QC items, aggregating the
per-item flags into final NRT flags, and — when the input already carries
NRT QC flags — comparing the existing and newly computed flags.
"""

from aiqclib.common.base.config_base import ConfigBase
from aiqclib.common.utils.progress import report_progress
from aiqclib.common.loader.nrtqc_loader import (
    load_nrtqc_step1_input_dataset,
    load_nrtqc_step2_qc_dataset,
    load_nrtqc_step3_concat_dataset,
    load_nrtqc_step4_compare_dataset,
)


def run_nrt_qc(config: ConfigBase, verbose: bool = False) -> None:
    """
    Execute the NRT QC pipeline for the given configuration.

    This function performs the following steps in sequence:
      1. Load and read the initial input data.
      2. Apply the configured QC items and write the per-item flag columns.
      3. Aggregate the item flags into a final NRT flag per variable
         (applying flag propagation items such as ``temp_to_psal`` last)
         and write the output parquet: the original input columns plus all
         QC item columns and the final NRT flags.
      4. When at least one variable has an existing flag column configured
         (its ``flag`` entry in the ``qc_variable_set``), build and write
         the per-variable flag comparison reports; otherwise the step is
         skipped.

    :param config: A configuration object specifying the classes and
                   parameters for each step of the NRT QC pipeline.
    :type config: ConfigBase
    :param verbose: When True, print each main step to stdout as it starts,
                    with the elapsed time of the run. Defaults to False.
    :type verbose: bool
    :return: None. The function performs I/O operations based on the
             configuration but does not return a value.
    :rtype: None
    """
    with report_progress(
        stage="nrt_qc",
        total_steps=4,
        enabled=verbose,
        label=getattr(config, "dataset_name", None),
    ) as progress:
        progress.step("Reading input data")
        ds_input = load_nrtqc_step1_input_dataset(config)
        ds_input.read_input_data()

        progress.step("Running QC items")
        ds_qc = load_nrtqc_step2_qc_dataset(config, ds_input.input_data)
        ds_qc.run_qc_items()
        ds_qc.write_qc_data()

        progress.step("Aggregating item flags into final NRT flags")
        ds_concat = load_nrtqc_step3_concat_dataset(config, ds_qc.qc_data)
        ds_concat.concat_flags()
        ds_concat.write_merged_data()

        comparable = any(
            config.get_variable_flag(target_name) is not None
            for target_name in config.get_target_names()
        )
        if comparable:
            progress.step("Comparing existing and new flags")
            ds_compare = load_nrtqc_step4_compare_dataset(config, ds_concat.merged_data)
            ds_compare.compare_targets()
            ds_compare.write_reports()
        else:
            progress.skip("Comparing existing and new flags")
