Step 3: Training & Evaluation
=============================

With a properly prepared dataset (from :doc:`./preparation`), you are now ready to train and evaluate a machine learning model. This workflow leverages the training, validation, and test sets created in the previous step to build a model, rigorously assess its performance using cross-validation, and generate final evaluation metrics on a held-out test set.

Like all workflows in ``aiqclib``, this process is controlled by a dedicated YAML configuration file, which, like the preparation config, utilizes the "building blocks" concept for modularity and reusability.

.. admonition:: Prerequisites

   This tutorial assumes you have successfully completed :doc:`./preparation`. The training process directly uses the output files (the split datasets) generated in that step. Ensure your ``~/aiqc_project/data/dataset_0001/training/`` directory exists and contains the prepared data.

The Training Workflow
---------------------

The training workflow follows a similar pattern to the preparation step: you will generate a new configuration template, customize it to define your model and validation strategy, point to your input data, and specify where the trained models should be saved.

Step 3.1: Generate the Configuration Template
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, use ``aiqclib`` to generate a boilerplate configuration template specifically for the training workflow.

.. code-block:: python

   import aiqclib as aq
   import os

   config_path = os.path.expanduser("~/aiqc_project/config/training_config.yaml")
   aq.write_config_template(
       file_name=config_path,
       stage="train"
   )

Step 3.2: Customize the Configuration File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now, open the newly created ``~/aiqc_project/config/training_config.yaml`` file in your text editor. Your primary goals are to define:

1.  **Input & Output Paths:** Where to find the prepared dataset and where to save the trained model.
2.  **Model & Validation Strategy:** Which machine learning model to train and what cross-validation method to use.

You will need to edit the ``path_info_sets``, ``step_class_sets``, ``step_param_sets``, and ``training_sets`` sections.

Before you modify the config, let's create a directory where your trained models will be saved:

.. code-block:: bash

   mkdir -p ~/aiqc_project/models

**Update your training_config.yaml file:**
Modify the file to align with the following structure. Remember to replace placeholder paths with your actual project setup.

.. note::
   ``aiqclib`` integrates multiple ML algorithms, and it is easy to switch between them. For more details, see the dedicated :doc:`../../how-to/algorithm_selection` page.

.. code-block:: yaml

    path_info_sets:
      - name: data_set_1
        common:
          base_path: ~/aiqc_project/data # Root directory of the prepared dataset (from preparation step)
        input:
          step_folder_name: training # Subdirectory containing the split training/validation/test data
        model:
          base_path: ~/aiqc_project/models # Directory where the final trained models will be saved

.. code-block:: yaml

    # Define your model and validation strategy here.
    # For this tutorial, we'll use a KFoldValidation and XGBoost model.
    step_class_sets:
      - name: training_step_set_1
        steps:
          input: InputTrainingSetA
          validate: KFoldValidation # Specify your cross-validation class
          model: XGBoost # Specify your ML model class (e.g., XGBoost, RandomForest)
          build: BuildModel

    # Define parameters for your chosen model and validation.
    # For example, number of folds for CV, or model hyperparameters.
    step_param_sets:
      - name: training_param_set_1
        steps:
          input: { }
          validate: { k_fold: 5 } # 5-fold cross-validation
          model: { calculate_shap: False,                   # Control SHAP value calculation
                   model_params: { scale_pos_weight: 200,   # Specify pos:neg ratio
                                   n_jobs: -1 } }           # Number of threads used by XGBoost
          build: { }

.. code-block:: yaml

    training_sets:
      - name: training_0001  # A unique name for this training job
        dataset_folder_name: dataset_0001  # This MUST match the dataset_folder_name from your preparation config
        path_info: data_set_1
        target_set: target_set_1 # This needs to match a 'target_set' defined in your prepare_config.yaml
        step_class_set: training_step_set_1
        step_param_set: training_param_set_1

.. note::
   The training configuration file includes many other options for advanced model selection, hyperparameter tuning, and cross-validation strategies. For a complete reference of all available parameters, please consult the dedicated :doc:`../../configuration/training` page.

Step 3.3: Run the Training Process
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once you have customized your ``training_config.yaml`` with the correct paths and model/validation configurations, you can execute the training and evaluation workflow.

Load the configuration file and then call the ``train_and_evaluate`` function:

.. code-block:: python

   import aiqclib as aq
   import os

   config_path = os.path.expanduser("~/aiqc_project/config/training_config.yaml")
   config = aq.read_config(config_path)
   aq.train_and_evaluate(config)

Understanding the Output
------------------------

After the command finishes, ``aiqclib`` will have created new folders within your dataset's output directory (e.g., ``~/aiqc_project/data/dataset_0001/``) and within your model's base path (``~/aiqc_project/models/``). The primary outputs include:

*   **validate**: Contains detailed results from the cross-validation process, allowing you to inspect model performance across different data folds. This includes metrics, predictions, and potentially visualizations.
*   **build**: Holds a comprehensive report of the final model's evaluation performance on the held-out test dataset, along with aggregated metrics.
*   **models**: Holds the final, trained model object(s) ready for classification. These are the artifacts you will use in the next step.

Next Steps
----------

You have now successfully trained and evaluated a machine learning model using ``aiqclib``! The final step in the workflow is to use this trained model to classify new, unseen data.

Proceed to the next tutorial: :doc:`./classification`.
