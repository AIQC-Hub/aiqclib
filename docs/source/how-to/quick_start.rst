Quick Start
=============================================

This guide demonstrates how to run the entire machine learning process with minimal configuration.

.. note::
   This is a condensed version of the tutorial provided in the "Getting Started" section. See the :doc:`../tutorial/overview` in the "Getting Started" section for more comprehensive explanations.

Objectives
-----------------------------

You will learn how to run all three stages of ``aiqclib`` by creating stage-specific configuration files. This guide lets you create three classifiers for ``temp`` (temperature), ``psal`` (salinity), and ``pres`` (pressure) to predict QC labels for the corresponding variables.

Installation
-----------------------------

(Optional) We recommend creating a ``mamba``/``conda`` environment before installing ``aiqclib``.

.. code-block:: bash

   # conda
   conda create --name aiqclib -c conda-forge python=3.12 pip uv
   conda activate aiqclib

   # mamba
   mamba create -n aiqclib -c conda-forge python=3.12 pip uv
   mamba activate aiqclib


Use ``pip``, ``conda``, or ``mamba`` to install ``aiqclib``.

.. code-block:: bash

   # pip
   pip install aiqclib

   # conda
   conda install -c conda-forge aiqclib

   # mamba
   mamba install -c conda-forge aiqclib


Download Raw Input Data
-----------------------------

You can get the sample input data set (``nrt_cora_bo_4.parquet``) from `Kaggle <https://www.kaggle.com/api/v1/datasets/download/takaya88/copernicus-marine-nrt-ctd-data-for-aiqc>`_.

Prepare Directory Structure
-----------------------------

The following Python commands create the necessary directory structure for your input and output files.

.. code-block:: python

    import os
    import polars as pl
    import aiqclib as aq

    print(f"aiqclib version: {aq.__version__}")

    # !! IMPORTANT: Update these placeholder paths to your actual file locations !!
    data_path = "/path/to/your/data"  # This will be the root for outputs

    config_path = os.path.join(data_path, "config")
    os.makedirs(config_path, exist_ok=True)

Stage 1: Data Preparation Stage
---------------------------------------------

The `prepare` workflow (`stage="prepare"`) is the first step in the machine learning pipeline. It processes your raw data into feature sets and then splits them into training, validation, and test sets.

Template Configuration File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following command creates a configuration template for this stage.

.. code-block:: python

    config_file_prepare = os.path.join(config_path, "data_preparation_config.yaml")
    aq.write_config_template(file_name=config_file_prepare, stage="prepare")

Update the Configuration File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**File:** ``/path/to/your/data/config/data_preparation_config.yaml``

1.  **Update Data and Input Paths:**
    Adjust the ``base_path`` values in the ``path_info_sets`` section.

    .. code-block:: yaml
       :caption: data_preparation_config.yaml: path_info_sets
       :emphasize-lines: 4, 6

       path_info_sets:
         - name: data_set_1
           common:
             base_path: /path/to/your/data  # <--- Root directory for generated datasets and models
           input:
             base_path: /path/to/your/input # <--- Directory where the raw input data is located
             step_folder_name: ""

2.  **Configure the Test Data Year(s):**
    Specify the year(s) to be held out as an independent test set. The ``remove_years`` parameter excludes these years from the training and validation sets.

    .. code-block:: yaml
       :caption: data_preparation_config.yaml: step_param_sets
       :emphasize-lines: 7

       step_param_sets:
         - name: data_set_param_set_1
           steps:
             input: { sub_steps: { rename_columns: false,
                                   filter_rows: true },
                      rename_dict: { },
                      filter_method_dict: { remove_years: [ 2023 ], # <--- Year(s) to set aside for the test set
                                            keep_years: [ ] } }

3.  **Specify Input File Name:**
    Ensure ``input_file_name`` matches the name of your raw data file.

    .. code-block:: yaml
       :caption: data_preparation_config.yaml: data_sets
       :emphasize-lines: 4

       data_sets:
         - name: dataset_0001
           dataset_folder_name: dataset_0001
           input_file_name: nrt_cora_bo_4.parquet # <--- Your input file's name

Run the Data Preparation Stage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once the configuration file is updated, run the following command to generate the training and validation datasets.

.. code-block:: python

    config_prepare = aq.read_config(os.path.join(config_path, "data_preparation_config.yaml"))
    aq.create_training_dataset(config_prepare)

Understanding the Output
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
After the command finishes, your main output directory (e.g., ``/path/to/your/data``) will contain a new folder named ``dataset_0001``. Inside this folder, you will find several subdirectories, each representing a stage of the data preparation pipeline:

*   **summary**: Contains intermediate files with summary statistics.
*   **select**: Stores data points identified as "good" (negative samples) and "bad" (positive samples).
*   **locate**: Contains specific observation records for positive and negative profiles.
*   **extract**: Holds the features extracted from the observation records.
*   **training**: The final output directory for this stage. It contains the split training, validation, and test datasets in Parquet format.

Stage 2: Training & Evaluation
--------------------------------

The `train` workflow (`stage="train"`) orchestrates the model building process. It uses the datasets from the `prepare` stage to perform cross-validation, train the model, and evaluate it.

Template Configuration File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following command creates a configuration template for this stage.

.. code-block:: python

    config_file_train = os.path.join(config_path, "training_config.yaml")
    aq.write_config_template(file_name=config_file_train, stage="train")

Update the Configuration File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**File:** ``/path/to/your/data/config/training_config.yaml``

1.  **Update Data Path:**
    Adjust the ``base_path`` in the ``path_info_sets`` section. This path must point to the same output directory (``common.base_path``) you defined in ``data_preparation_config.yaml``.

    .. code-block:: yaml
       :caption: training_config.yaml: path_info_sets
       :emphasize-lines: 4

       path_info_sets:
         - name: data_set_1
           common:
             base_path: /path/to/your/data # <--- Must match the common.base_path from the previous stage

Run the Training & Evaluation Stage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With the configuration file updated, the following command will run the training and validation processes.

.. code-block:: python

    config_train = aq.read_config(os.path.join(config_path, "training_config.yaml"))
    aq.train_and_evaluate(config_train)

Understanding the Output
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After the command finishes, new folders will be created within your dataset's output directory (e.g., ``/path/to/your/data/dataset_0001/``). The primary outputs include:

*   **validate**: Contains detailed results from the cross-validation process, allowing you to inspect model performance across different data folds.
*   **build**: Holds a comprehensive report of the final model's evaluation on the held-out test dataset.
*   **model**: Contains the final, trained model objects. These are the artifacts you will use in the next stage.

Stage 3: Classification
-----------------------------

The `classify` workflow (`stage="classify"`) applies a trained model to make predictions on a new, unseen dataset (e.g., the test set you held out in Stage 1).

Template Configuration File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following command creates a configuration template for this final stage.

.. code-block:: python

    config_file_classify = os.path.join(config_path, "classification_config.yaml")
    aq.write_config_template(file_name=config_file_classify, stage="classify")

Update the Configuration File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**File:** ``/path/to/your/data/config/classification_config.yaml``

1.  **Update Paths:**
    Adjust the ``base_path`` values for ``common``, ``input``, and ``model``.
    *   ``common.base_path``: The root directory for your data outputs.
    *   ``input.base_path``: The location of the raw input data file.
    *   ``model.base_path``: The location of the trained model from Stage 2.

    .. code-block:: yaml
       :caption: classification_config.yaml: path_info_sets
       :emphasize-lines: 4, 6, 9

       path_info_sets:
         - name: data_set_1
           common:
             base_path: /path/to/your/data  # <--- Your common data root
           input:
             base_path: /path/to/your/input # <--- Location of the raw data for classification
             step_folder_name: ""
           model:
             base_path: /path/to/your/data/dataset_0001 # <--- Path to the trained model folder
             step_folder_name: "model"

2.  **Configure Classification Data Year(s):**
    Specify the year(s) for the classification dataset using ``keep_years``. This should correspond to the test data year(s) you excluded (``remove_years``) during data preparation.

    .. code-block:: yaml
       :caption: classification_config.yaml: step_param_sets
       :emphasize-lines: 8

       step_param_sets:
         - name: data_set_param_set_1
           steps:
             input: { sub_steps: { rename_columns: false,
                                   filter_rows: true },
                      rename_dict: { },
                      filter_method_dict: { remove_years: [],
                                            keep_years: [ 2023 ] } } # <--- Specify year(s) to *keep* for classification

3.  **Specify Input File Name:**
    Ensure ``input_file_name`` matches the name of the data file you want to classify.

    .. code-block:: yaml
       :caption: classification_config.yaml: data_sets
       :emphasize-lines: 4

       data_sets:
         - name: classification_0001
           dataset_folder_name: dataset_0001
           input_file_name: nrt_cora_bo_4.parquet # <--- Your input file's name

Run the Classification Stage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once the configuration is complete, the following commands will apply the model to the specified data and generate classification results.

.. code-block:: python

    config_classify = aq.read_config(os.path.join(config_path, "classification_config.yaml"))
    aq.classify_dataset(config_classify)

Understanding the Output
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After this command finishes, the output directories will be generated within ``/path/to/your/data/dataset_0001/``. The most important output is in the ``classify`` directory:

*   **classify**: This is the final output directory for the workflow. It contains:

    *   A ``.parquet`` file with the original input data augmented with new columns for the model's predictions (e.g., ``temp_prediction``) and prediction probabilities (e.g., ``temp_probability``).
    *   A summary report detailing the classification results.

Other intermediate folders (``summary``, ``select``, ``locate``, ``extract``) are also created, mirroring the process used during data preparation to ensure consistency.

Conclusion
--------------

Congratulations! You have successfully completed the entire ``aiqclib`` workflow, from raw data preparation to training a machine learning model and using it to generate predictions on new data.

You now have a powerful, repeatable, and configurable pipeline for your machine learning tasks. You can easily adapt the configuration files to process new datasets, experiment with different models, or integrate this pipeline into larger automated workflows.