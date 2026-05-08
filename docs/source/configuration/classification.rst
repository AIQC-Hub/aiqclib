Configuration of Classification
=================================
The ``classify`` workflow (``stage="classify"``) is designed to apply a pre-trained machine learning model to new, unseen datasets to generate predictions. It leverages the same modular "building blocks" concept found in the ``prepare`` and ``train`` workflows, but its configuration is streamlined.

Unlike preparation or training, classification doesn't involve model learning or complex feature engineering from scratch. Instead, its primary role is to orchestrate the application of an existing model. This involves:

*   Loading a pre-trained model.
*   Applying the same feature engineering transformations (using the same parameters) as were used during training.
*   Making predictions on new input data.
*   Saving the classified output.

You will typically need to configure ``path_info_sets`` and ``classification_sets``, along with ensuring consistency for ``target_sets``, ``summary_stats_sets``, and ``feature_sets`` with the model's training context.

Key Configuration Sections
--------------------------

`path_info_sets`
^^^^^^^^^^^^^^^^
This section defines all the necessary file system locations for the classification workflow. It tells ``aiqclib`` where to find the raw input data, the trained model, and where to save the final classified output.

*   **common.base_path**: The root directory where all outputs (intermediate files, final classified data) for this classification task will be saved.
*   **input.base_path**: The directory where your raw input files (the data you want to classify) are located.
*   **model.base_path**: The path to the directory containing your trained model files from the ``train`` workflow.
*   **concat.step_folder_name**: The name of the subdirectory within ``common.base_path`` where the final combined/classified output file will be stored.

.. code-block:: yaml
   :caption: Example path_info_sets
   :name: classify-path-info-sets

   path_info_sets:
     - name: data_set_1
       common:
         base_path: /path/to/data
       input:
         base_path: /path/to/input
         step_folder_name: ""
       model:
         base_path: /path/to/data/dataset_0001
         step_folder_name: "model"
       concat:
         step_folder_name: classify

`target_sets`
^^^^^^^^^^^^^
While the classification workflow is applying a pre-trained model, this section is still important for consistency. It specifies the target variables and their associated quality control (QC) flags. This ensures that any initial data filtering or subsequent evaluation/post-processing steps properly recognize the columns that the model was trained to predict. The definitions here should align with those used during the model's training.

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

`feature_sets` & `feature_param_sets`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(**Advanced Use**)
These two sections are critical for ensuring that the new input data is transformed into the exact same feature space that the pre-trained model expects. They should explicitly list the feature engineering methods and their parameters that were applied during the model's training phase.

*   **feature_sets**: Lists the names of the feature engineering methods to re-apply to the new input data.
*   **feature_param_sets**: Provides the specific parameters for each feature method listed in ``feature_sets``. These parameters (e.g., ``stats_set`` references for normalization) are usually copied directly from the ``prepare_config.yaml`` used for training.

.. code-block:: yaml

   # A list of features to apply for classification
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
           convert: cosine
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
This section defines the specific Python classes that implement the logic for each step in the classification pipeline. While ``aiqclib`` provides default implementations, this allows for customization of how data is ingested, features are generated, the model is loaded, predictions are made, and results are output.

*   **steps.input**: Class for handling initial data loading.
*   **steps.summary**: Class for processing summary statistics (if applicable).
*   **steps.select**: Class for selecting specific data points.
*   **steps.locate**: Class for spatial or temporal localization (if features depend on neighbors).
*   **steps.extract**: Class for extracting features based on the ``feature_sets`` configuration.
*   **steps.model**: Class for loading the pre-trained machine learning model.
*   **steps.classify**: Class for performing the actual prediction using the loaded model.
*   **steps.concat**: Class for concatenating or combining the final classified results.

.. code-block:: yaml

   step_class_sets:
     - name: data_set_step_set_1
       steps:
         input: InputDataSetAll
         summary: SummaryDataSetAll
         select: SelectDataSetAll
         locate: LocateDataSetAll
         extract: ExtractDataSetAll
         model: XGBoost
         classify: ClassifyAll
         concat: ConcatDataSetAll

`step_param_sets`
^^^^^^^^^^^^^^^^^
This section provides general parameters for the workflow processes defined in ``step_class_sets``. These parameters control the behavior of various pipeline steps during classification.

*   **steps.input.sub_steps.filter_rows**: A boolean flag to enable or disable row filtering based on ``filter_method_dict``.
*   **steps.input.filter_method_dict.keep_years**: Specifies a list of years from which data should be kept for classification. Other years will be excluded.
*   **steps.input.rename_dict**: Dictionary for renaming columns during input processing.
*   **steps.model.calculate_shap**: This is used to control SHAP value calculation.
*   **steps.model.model_params.n_jobs**: The number of threads used by XGBoost. It tries to use all available CPU cores if it is set to `-1`.
*   Parameters for other steps (``summary``, ``select``, ``locate``, ``extract``, ``classify``, ``concat``) are also defined here, often left empty if default behavior is sufficient or if parameters are handled by the model itself.

.. code-block:: yaml

   step_param_sets:
     - name: data_set_param_set_1
       steps:
         input: { sub_steps: { rename_columns: false,
                               filter_rows: true },
                  rename_dict: { },
                  filter_method_dict: { remove_years: [],
                                        keep_years: [2023] } }
         summary: { }
         select: { }
         locate: { }
         extract: { }
         model: { calculate_shap: False,
                  model_params: { n_jobs: -1 } }
         classify: { }
         concat: { }

`classification_sets`
^^^^^^^^^^^^^^^^^^^^^
This is the main "assembly" section that defines a complete classification job. Each entry in this list orchestrates a unique classification run by linking together the input data file with the specific path, target variable, feature engineering, and pipeline step configurations.

*   **name**: A unique identifier for this classification task (e.g., "NRT_BO_001").
*   **dataset_folder_name**: The name of the folder within ``common.base_path`` where intermediate and final classified results specific to this job will be stored. This often matches the name used during preparation and training to maintain consistency.
*   **input_file_name**: The name of the raw data file (e.g., a ``.parquet`` file) that you want to classify. This file should be located in ``input.base_path``.
*   **path_info**: The ``name`` of the path configuration to use from ``path_info_sets``.
*   **target_set**: The ``name`` of the target variable configuration to use from ``target_sets``.
*   ...and similarly for all other configuration sets.

.. code-block:: yaml
   :caption: Example classification_sets
   :name: classify-sets

   classification_sets:
     - name: classification_0001
       dataset_folder_name: dataset_0001
       input_file_name: nrt_cora_bo_4.parquet
       path_info: data_set_1
       target_set: target_set_1
       # ... other set references would follow here

.. note::
   While you can define multiple classification sets in the ``classification_sets`` section, a specific one must be selected for subsequent processes. Please consult the dedicated :doc:`../../how-to/selecting_specific_configurations` page for instructions on how to do this.

Full Example with XGBoost
--------------------------------

Here is a complete example of a ``classification_config.yaml`` file, showing how all the building blocks come together. The lines you will most commonly need to edit or customize are highlighted for quick reference.

.. code-block:: yaml
   :caption: Full classification_config.yaml example
   :emphasize-lines: 5, 7, 10, 11, 13, 68, 72, 95, 97, 98, 103, 104

   ---
   path_info_sets:
     - name: data_set_1
       common:
         base_path: /path/to/data # Root output directory for processed data
       input:
         base_path: /path/to/input # Directory with raw input files
         step_folder_name: ""
       model:
         base_path: /path/to/model  # Directory containing trained model files
         step_folder_name: "model"  # Change it to "" if you like to avoid /path/to/model/model/
       concat:
         step_folder_name: classify # Subdirectory for final classification results

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
           convert: cosine
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
         input: InputDataSetAll
         summary: SummaryDataSetAll
         select: SelectDataSetAll
         locate: LocateDataSetAll
         extract: ExtractDataSetAll
         model: XGBoost
         classify: ClassifyAll
         concat: ConcatDataSetAll

   step_param_sets:
     - name: data_set_param_set_1
       steps:
         input: { sub_steps: { rename_columns: false,
                               filter_rows: true },
                  rename_dict: { },
                  filter_method_dict: { remove_years: [],
                                        keep_years: [2023] } }
         summary: { }
         select: { }
         locate: { }
         extract: { }
         model: { calculate_shap: False,
                  model_params: { n_jobs: -1 } }
         classify: { }
         concat: { }

   classification_sets:
     - name: classification_0001  # A unique name for this classification task
       dataset_folder_name: dataset_0001  # Folder name for intermediate/output files for this job
       input_file_name: nrt_cora_bo_4.parquet   # The raw input filename to classify
       path_info: data_set_1
       target_set: target_set_1
       summary_stats_set: summary_stats_set_1
       feature_set: feature_set_1
       feature_param_set: feature_set_1_param_set_1
       feature_stats_set: feature_set_1_stats_set_1
       step_class_set: data_set_step_set_1
       step_param_set: data_set_param_set_1
