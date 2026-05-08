Configuration of Dataset Preparation
======================================
The ``prepare`` workflow (``stage="prepare"``) is central to setting up your data for machine learning tasks within this library. It provides comprehensive control over the entire data processing pipeline, from  preparing feature data sets from your raw data and creating the training, validation, and test data sets.

Core Concepts: Modular Configuration
------------------------------------
The configuration for dataset preparation is designed around a powerful "building blocks" concept. Instead of defining a monolithic configuration, you define various sets of specialized configurations once, give each set a unique name, and then combine them as needed to construct a complete and flexible data processing pipeline. This modularity promotes reusability, simplifies experimentation, and enhances maintainability.

The primary configuration sections (building blocks) are:

*   **path_info_sets**: Defines reusable directory structures for input data and processed outputs.
*   **target_sets**: Specifies the prediction target variables, including their quality control (QC) flags.
*   **summary_stats_sets**: Configures summary statistics.
*   **feature_sets**: (**Advanced**) Lists the specific feature engineering methods to be applied.
*   **feature_param_sets**: Provides detailed parameters and settings for each chosen feature engineering method.
*   **feature_stats_sets**: (**Advanced**) Provides summary statistics values for normalizing features.
*   **step_class_sets**: (**Advanced**) Allows users to define custom Python classes for individual processing steps, enabling deep customization of the pipeline's behavior.
*   **step_param_sets**: Supplies general parameters that control the behavior of the default or custom processing steps.
*   **data_sets**: The central assembly section, where you combine named blocks from the sections above to define a complete and executable data processing pipeline.

.. note::

   ``dmaclib`` provides methods to down-sample the negative data set. Please refer to the :doc:`../../how-to/down_sampling_negative` guide for details.

Detailed Configuration Sections
-------------------------------

`path_info_sets`
^^^^^^^^^^^^^^^^
This section defines the critical file system locations for both your raw input data and the various processed output artifacts. You can define multiple named path configurations to easily switch between different storage environments or project setups.

*   **common.base_path**: The root directory where all processed data and intermediate artifacts will be saved by this workflow.
*   **input.base_path**: The directory containing your raw input data files.
*   **split.step_folder_name**: The name of the subdirectory where the final training, validation, and test datasets will be stored (e.g., `training`).

.. code-block:: yaml

   path_info_sets:
     - name: data_set_1
       common:
         base_path: /path/to/data
       input:
         base_path: /path/to/input
         step_folder_name: ""
       split:
         step_folder_name: training

`target_sets`
^^^^^^^^^^^^^
This section specifies the target variables that your machine learning model will predict. For each target variable, you must also define its corresponding quality control (QC) flag column. These flags are crucial for identifying good versus bad data points, allowing the pipeline to filter or weight data appropriately. You define both positive (good) and negative (bad) flag values.

.. code-block:: yaml

   target_sets:
     - name: target_set_1
       variables:
         - name: temp
           flag: temp_qc
           pos_flag_values: [ 4, 6, 7 ]
           neg_flag_values: [ 1 ]

`summary_stats_sets`
^^^^^^^^^^^^^^^^^^^^
This section defines summary statistics that will be used for feature values or feature normalization.

.. code-block:: yaml

   summary_stats_sets:
     - name: summary_stats_set_1
       stats:
         - name: location
           col_names: [ longitude, latitude ]
         - name: profile_summary_stats
           col_names: [ temp, psal, pres ]
         - name: basic_values3
           col_names: [ temp, psal, pres ]

``aiqclib`` currently provides the following summary statistics.

*   **location**: global summary statistics of locations for feature normalization.
*   **profile_summary_stats**: profile level summary statistics used as features and for feature normalization.
*   **basic_values3**: global summary statistics of specified variables for feature normalization.

`feature_sets` & `feature_param_sets`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
These two interconnected sections are dedicated to configuring your feature engineering process.

*   **feature_sets**: This block lists the *names* of the specific feature engineering methods you want to apply to your data.
*   **feature_param_sets**: This block provides the detailed parameters and configurations for each of the feature methods listed in your chosen ``feature_sets`` block. This allows for fine-grained control over how each feature is generated.

.. code-block:: yaml

   # A list of features to apply
   feature_sets:
     - name: feature_set_1
       features:
         - location
         - day_of_year
         - profile_summary_stats
         - basic_values
         - flank_up
         - flank_down

   # Parameters for the features listed above
   feature_param_sets:
     - name: feature_set_1_param_set_1
       params:
         - feature: location
           stats_set: { type: raw }
           col_names: [ longitude, latitude ]
         - feature: day_of_year
           convert: cosine                         # or sine
           col_names: [ profile_timestamp ]
         - feature: profile_summary_stats
           stats_set: { type: raw }
           col_names: [ temp, psal, pres ]
           summary_stats_names: [ mean, median, sd, pct25, pct75 ]
         - feature: basic_values
           stats_set: { type: raw }
           col_names: [ temp, psal, pres ]
         - feature: flank_up
           flank_up: 5
           stats_set: { type: raw }
           col_names: [ temp, psal, pres ]
         - feature: flank_down
           flank_down: 5
           stats_set: { type: raw }
           col_names: [ temp, psal, pres ]

`feature_stats_sets`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(**Advanced Use**)

This section defines summary statistics that will be used for normalization or scaling of feature values. These statistics are typically derived from your dataset itself to ensure proper scaling.

.. code-block:: yaml

   feature_stats_sets:
     - name: feature_set_1_stats_set_1

.. important::

   As it is crucial to normalize features for non-tree based machine learning methods, such as SVM and logistic regression, you need to provide summary statistics (like min/max values) of your data in the configuration file. The ``aiqclib`` library offers convenient functions to calculate the summary statistics.  Please refer to the :doc:`../../how-to/feature_normalization` guide for details.

`step_class_sets`
^^^^^^^^^^^^^^^^^
(**Advanced Use**)
This section allows you to define and reference custom Python classes that implement the logic for specific processing steps within the data preparation pipeline. While the library provides default implementations for all steps, this block gives advanced users the flexibility to replace or extend pipeline behaviors with their own code. Each entry maps a step name (e.g., ``input``, ``summary``) to the name of a Python class.

.. code-block:: yaml

   step_class_sets:
     - name: data_set_step_set_1
       steps:
         input: InputDataSetA
         summary: SummaryDataSetA
         select: SelectDataSetAll
         locate: LocateDataSetAll
         extract: ExtractDataSetA
         split: SplitDataSetAll

`step_param_sets`
^^^^^^^^^^^^^^^^^
This section provides general parameters that control the behavior of the various data processing steps within the pipeline (whether default or custom ``step_class_sets``). Examples of parameters include data filtering rules, sampling ratios, and split configurations.

*   **steps.input.sub_steps.filter_rows**: A boolean flag to enable/disable row filtering based on ``filter_method_dict``.
*   **steps.input.filter_method_dict.remove_years**: Specifies a list of years to be excluded from the dataset.
*   **steps.input.filter_method_dict.keep_years**: Specifies a list of years to be kept for training.
*   **steps.split.test_set_fraction**: Defines the proportion of data to allocate to the test set.
*   **steps.split.k_fold**: Defines the `k` of k-fold cross validation

.. code-block:: yaml

   step_param_sets:
     - name: data_set_param_set_1
       steps:
         input: { sub_steps: { rename_columns: false,
                               filter_rows: true },
                  rename_dict: { },
                  filter_method_dict: { remove_years: [2023],
                                        keep_years: [] } }
         summary: { }
         select: { }
         locate: { }
         extract: { }
         split: { test_set_fraction: 0.1,
                  k_fold: 5 }

`data_sets`
^^^^^^^^^^^
This is the main "pipeline assembly" section. Each entry in this list defines a complete data preparation job by linking together the named building blocks defined in the other sections. This section essentially orchestrates which specific configuration sets are used for a given dataset processing run.

*   **name**: A unique identifier for this particular dataset preparation job (e.g., ``dataset_0001``).
*   **dataset_folder_name**: The name of the specific folder that will be created within the ``common.base_path`` to store outputs for this job (e.g., ``dataset_0001``).
*   **input_file_name**: The specific raw data file (located in ``input.base_path``) to be processed for this job.
*   **path_info**: The ``name`` of the path configuration to use from ``path_info_sets``.
*   **target_set**: The ``name`` of the target configuration to use from ``target_sets``.
*   ...and similarly for all other configuration sets.

.. code-block:: yaml

   data_sets:
     - name: dataset_0001
       dataset_folder_name: dataset_0001
       input_file_name: nrt_cora_bo_4.parquet
       path_info: data_set_1
       target_set: target_set_1
       # ... other set references would follow here

.. note::
   While you can define multiple data sets in the ``data_sets`` section, a specific one must be selected for subsequent processes. Please consult the dedicated :doc:`../../how-to/selecting_specific_configurations` page for instructions on how to do this.

Full Example
------------

Below is a complete example of a ``prepare_config.yaml`` file, demonstrating how all the building blocks are combined. The lines you will most commonly need to edit or customize are highlighted for quick reference.

.. code-block:: yaml
   :caption: Full prepare_config.yaml example
   :emphasize-lines: 5, 7, 65, 69, 90, 92, 93, 98, 99, 102, 103, 104

   ---
   path_info_sets:
     - name: data_set_1
       common:
         base_path: /path/to/data # Root output directory for processed data
       input:
         base_path: /path/to/input # Directory containing raw input files
         step_folder_name: ""
       split:
         step_folder_name: training

   target_sets:
     - name: target_set_1
       variables:
         - name: temp
           flag: temp_qc
           pos_flag_values: [ 4, 6, 7 ]
           neg_flag_values: [ 1 ]
         - name: psal
           flag: psal_qc
           pos_flag_values: [ 4, 6, 7 ]
           neg_flag_values: [ 1 ]
         - name: pres
           flag: pres_qc
           pos_flag_values: [ 4, 6, 7 ]
           neg_flag_values: [ 1 ]

   summary_stats_sets:
     - name: summary_stats_set_1
       stats:
         - name: location
           col_names: [ longitude, latitude ]
         - name: profile_summary_stats
           col_names: [ temp, psal, pres ]
         - name: basic_values3
           col_names: [ temp, psal, pres ]

   feature_sets:
     - name: feature_set_1
       features:
         - location
         - day_of_year
         - profile_summary_stats
         - basic_values
         - flank_up
         - flank_down

   feature_param_sets:
     - name: feature_set_1_param_set_1
       params:
         - feature: location
           stats_set: { type: raw }
           col_names: [ longitude, latitude ]
         - feature: day_of_year
           convert: cosine                         # or sine
           col_names: [ profile_timestamp ]
         - feature: profile_summary_stats
           stats_set: { type: raw }
           col_names: [ temp, psal, pres ]
           summary_stats_names: [ mean, median, sd, pct25, pct75 ]
         - feature: basic_values
           stats_set: { type: raw }
           col_names: [ temp, psal, pres ]
         - feature: flank_up
           flank_up: 5
           stats_set: { type: raw }
           col_names: [ temp, psal, pres ]
         - feature: flank_down
           flank_down: 5
           stats_set: { type: raw }
           col_names: [ temp, psal, pres ]

   feature_stats_sets:
     - name: feature_set_1_stats_set_1

   step_class_sets:
     - name: data_set_step_set_1
       steps:
         input: InputDataSetA
         summary: SummaryDataSetA
         select: SelectDataSetAll
         locate: LocateDataSetAll
         extract: ExtractDataSetA
         split: SplitDataSetAll

   step_param_sets:
     - name: data_set_param_set_1
       steps:
         input: { sub_steps: { rename_columns: false,
                               filter_rows: true },
                  rename_dict: { },
                  filter_method_dict: { remove_years: [2023],
                                        keep_years: [] } }
         summary: { }
         select: { }
         locate: { }
         extract: { }
         split: { test_set_fraction: 0.1,
                  k_fold: 5 }

   data_sets:
     - name: dataset_0001  # Your unique name for this dataset job
       dataset_folder_name: dataset_0001  # The folder name for output files
       input_file_name: nrt_cora_bo_4.parquet # The specific raw input file to process
       path_info: data_set_1
       target_set: target_set_1
       summary_stats_set: summary_stats_set_1
       feature_set: feature_set_1
       feature_param_set: feature_set_1_param_set_1
       feature_stats_set: feature_set_1_stats_set_1
       step_class_set: data_set_step_set_1
       step_param_set: data_set_param_set_1
