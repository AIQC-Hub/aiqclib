Configuration of Training & Evaluation
========================================
The ``train`` workflow (``stage="train"``) is responsible for orchestrating the machine learning model building process. It takes the prepared dataset (the output from the ``prepare`` workflow) and handles critical steps such as cross-validation, actual model training, and final evaluation on a held-out test set.

While the ``prepare`` workflow focuses on complex data transformation and feature engineering, the ``train`` configuration is generally simpler. Its primary role is to leverage the "building blocks" concept to specify:

*   The machine learning model to be used.
*   The chosen validation strategy (e.g., k-fold cross-validation).
*   The locations of the prepared input data and where to save the final trained models.

Detailed Configuration Sections
-------------------------------

`path_info_sets`
^^^^^^^^^^^^^^^^
This section is crucial for linking the training workflow to the prepared datasets from the previous ``prepare`` stage and defining where to save the resulting trained models.

*   **common.base_path**: The root directory where the prepared dataset (output from the ``prepare`` workflow) is located. This typically corresponds to the ``common.base_path`` defined in your ``prepare_config.yaml``.
*   **input.step_folder_name**: The name of the subdirectory within the prepared dataset's folder where the final training/validation/test splits are located (e.g., ``training``).

.. code-block:: yaml

   path_info_sets:
     - name: data_set_1
       common:
         base_path: /path/to/data
       input:
         step_folder_name: training

`target_sets`
^^^^^^^^^^^^^
Similar to the ``prepare`` workflow, this section specifies the target variables for your machine learning model. It ensures that the training process correctly identifies which column represents the prediction target and understands its associated quality control (QC) flags, which are often used to filter or weight data during training.

.. code-block:: yaml

   target_sets:
     - name: target_set_1
       variables:
         - name: temp
           flag: temp_qc
           pos_flag_values: [ 4, 6, 7  ]
           neg_flag_values: [ 1 ]

`step_class_sets`
^^^^^^^^^^^^^^^^^
This powerful section allows you to define the core components of your training pipeline by specifying the Python classes to use for each major step. This is where you choose your machine learning model, the cross-validation method, and other pipeline components.

*   **steps.input**: The class responsible for ingesting the prepared training, validation, and test datasets.
*   **steps.validate**: The class defining the cross-validation strategy (e.g., ``KFoldValidation``, ``TimeSeriesValidation``).
*   **steps.model**: The class for the machine learning algorithm to be trained (e.g., ``XGBoost``, ``RandomForest``).
*   **steps.build**: The class that handles the final model training on the full training set and saving the model artifacts.

.. code-block:: yaml

   step_class_sets:
     - name: training_step_set_1
       steps:
         input: InputTrainingSetA
         validate: KFoldValidation
         model: XGBoost
         build: BuildModel

.. note::
   ``aiqclib`` integrates multiple ML algorithms, and it is easy to switch between them by setting the ``model`` key. For more details, see the dedicated :doc:`../../how-to/algorithm_selection` page.

`step_param_sets`
^^^^^^^^^^^^^^^^^
This section provides detailed parameters for the classes defined in your chosen ``step_class_sets``. This allows you to fine-tune the behavior of each step, such as specifying the number of folds for cross-validation or providing hyperparameters for your machine learning model.

*   **steps.input**: Parameters for the input data loading step (often empty or simple flags).
*   **steps.validate.k_fold**: For ``KFoldValidation``, specifies the number of folds for cross-validation.
*   **steps.model.calculate_shap**: This is used to control SHAP value calculation.
*   **steps.model.model_params.scale_pos_weight**: This is used to address imbalanced datasets by weighting the positive class. For example, ``200`` indicates a ratio of negative to positive records of 200:1.
*   **steps.model.model_params.n_jobs**: The number of threads used by XGBoost. It tries to use all available CPU cores if it is set to `-1`.
*   **steps.build**: Parameters for the final model building step (often empty or simple flags for saving).

.. code-block:: yaml

   step_param_sets:
     - name: training_param_set_1
       steps:
         input: { }
         validate: { k_fold: 5 }
         model: { calculate_shap: False,
                  model_params: { scale_pos_weight: 200,
                                  n_jobs: -1 } }
         build: { }

.. note::
   SHAP values can be automatically calculated during the test phase. For more details, see the dedicated :doc:`../../how-to/shap_values` page.

.. note::
   Model parameters specified by ``model_params`` differ across ML algorithms. Please consult the dedicated :doc:`../../how-to/algorithm_selection` page.

`training_sets`
^^^^^^^^^^^^^^^^^
This is the main "assembly" section that defines a complete training and evaluation job. Each entry in this list orchestrates a unique training run by linking together the prepared dataset with the specific path, target variable, and step configurations (classes and parameters).

*   **name**: A unique identifier for this particular training job.
*   **dataset_folder_name**: The name of the specific folder (created by the ``prepare`` workflow) containing the prepared data for this job (e.g., ``dataset_0001``).
*   **path_info**: The ``name`` of the path configuration to use from ``path_info_sets``.
*   **target_set**: The ``name`` of the target variable configuration to use from ``target_sets``.
*   **step_class_set** & **step_param_set**: The ``name`` of the step class and parameter configurations to use, respectively.

.. code-block:: yaml

   training_sets:
     - name: training_0001
       dataset_folder_name: dataset_0001
       path_info: data_set_1
       target_set: target_set_1
       step_class_set: training_step_set_1
       step_param_set: training_param_set_1

.. note::
   While you can define multiple training sets in the ``training_sets`` section, a specific one must be selected for subsequent processes. Please consult the dedicated :doc:`../../how-to/selecting_specific_configurations` page for instructions on how to do this.

Full Example with XGBoost
---------------------------

Below is a complete example of a ``training_config.yaml`` file. The lines you will most commonly need to edit or customize are highlighted for quick reference.

.. code-block:: yaml
   :caption: Full training_config.yaml example
   :emphasize-lines: 5, 38, 39, 40, 44, 45

   ---
   path_info_sets:
     - name: data_set_1
       common:
         base_path: /path/to/data # Root directory containing prepared data
       input:
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

   step_class_sets:
     - name: training_step_set_1
       steps:
         input: InputTrainingSetA
         validate: KFoldValidation
         model: XGBoost
         build: BuildModel

   step_param_sets:
     - name: training_param_set_1
       steps:
         input: { }
         validate: { k_fold: 5 }
         model: { calculate_shap: False,
                  model_params: { scale_pos_weight: 200,
                                  n_jobs: -1 } }
         build: { }

   training_sets:
     - name: training_0001  # A unique name for this training job
       dataset_folder_name: dataset_0001  # The folder name containing the prepared data for this job
       path_info: data_set_1
       target_set: target_set_1
       step_class_set: training_step_set_1
       step_param_set: training_param_set_1
