Step 3: Dataset Preparation
===========================

The dataset preparation workflow is the first crucial step in the ``aiqclib`` pipeline. It's designed to prepare feature data sets from your raw data. This includes creating training, validation, and test data sets.

This tutorial starts from the input Parquet file produced in
:doc:`./input_data`, either the downloaded example
(``nrt_cora_bo_4.parquet``) or one you generated yourself.

This entire process is driven by a YAML configuration file, ensuring your data preparation is repeatable, transparent, and easy to manage across different experiments.

.. admonition:: A Note on Running the Examples

   The examples in these tutorials are presented as commands suitable for an interactive Python session (e.g., in a terminal with ``python`` or ``ipython``, or within a Jupyter Notebook/Lab).

   However, you are encouraged to use the method you are most comfortable with. The code can be run in several ways:

   *   **In an Interactive Python Session:** Launch Python (``python``) or IPython (``ipython``) and paste the code line by line. This is great for quick tests and exploration.
   *   **As Python Scripts:** Copy the code into a ``.py`` file (e.g., ``prepare_data.py``) and execute it from your terminal with ``python your_script_name.py``. This is suitable for automation and batch processing.
   *   **In a Jupyter Notebook or Lab:** This is a fantastic option for experimentation, as it allows you to run code in cells, add notes, and visualize results interactively.

   Feel free to adapt the examples to your preferred environment.

The Dataset Preparation Workflow
--------------------------------
The ``aiqclib`` data preparation workflow consists of three main programmatic steps: generating a configuration template, customizing this template to match your data and desired processing, and finally running the preparation script.

Step 3.1: Generate the Configuration Template
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
First, use ``aiqclib`` to generate a boilerplate configuration template. This file will contain all the necessary sections for the data preparation task, which you will then customize.

.. code-block:: python

   import aiqclib as aq

   config_path = "~/aiqc_project/config/prepare_config.yaml"
   aq.write_config_template(
       file_name=config_path,
       stage="prepare"
   )

.. note::

   The output directory must already exist — writing into a missing directory
   is refused so that a mistyped path is reported instead of silently creating
   folders. If you skipped the directory setup in :doc:`./input_data`, pass
   ``create_dirs=True`` to create it as part of the call. A leading ``~`` is
   expanded to your home directory, both here and in the ``base_path`` values
   inside the configuration file.

   An existing file is never replaced either: re-running this after you have
   customized the config raises ``FileExistsError`` rather than resetting your
   edits. Pass ``overwrite=True`` when a fresh template is what you want.

Step 3.2: Customize the Configuration File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Now, open the newly created ``~/aiqc_project/config/prepare_config.yaml`` in a text editor. You need to tell ``aiqclib`` where to find your input data, where to save the processed output, and define your targets and features.

You will primarily focus on updating the following sections:

*   **path_info_sets**: Define your input and output directories.
*   **target_sets**: Specify your prediction targets and their quality control
    flags. The template labels from ``temp_qc`` / ``psal_qc``, which are
    near-real-time QC flags rather than delayed-mode ones — see
    :ref:`which-qc-flags` for what that means for your model.
*   **summary_stats_sets**: Provide settings for summary statistics.
*   **feature_sets & feature_param_sets**: List the feature engineering methods and their parameters.
*   **feature_stats_sets**: Provide statistics for feature normalization.
*   **data_sets**: Assemble the full pipeline by linking the named blocks.

**Updating path_info_sets and data_sets:**
Update your ``prepare_config.yaml`` to match the following for the ``path_info_sets`` and ``data_sets`` sections, replacing the placeholder paths with the ones you created in :doc:`./input_data`.

.. code-block:: yaml

    path_info_sets:
      - name: data_set_1
        common:
          base_path: ~/aiqc_project/data # Root directory for all processed output data
        input:
          base_path: ~/aiqc_project/input # Directory where your raw input files are located
          step_folder_name: "" # Set to "" if input files are directly in base_path
        split:
          step_folder_name: training # Subdirectory for the final training/validation/test splits

.. code-block:: yaml

    data_sets:
      - name: dataset_0001  # A unique name for this dataset preparation job
        dataset_folder_name: dataset_0001  # The name of the output folder for this job
        input_file_name: nrt_cora_bo_4.parquet # The specific raw input file to process

.. note::
   The ``prepare_config.yaml`` can be quite detailed. For a complete reference of all available configuration options, please consult the dedicated :doc:`../configuration/preparation` page.

.. note::

   ``aiqclib`` provides methods to down-sample the negative data set. Please refer to the :doc:`../how-to/down_sampling_negative` guide for details.

Step 3.3: Run the Preparation Process
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Once you have customized your ``prepare_config.yaml`` with the correct paths, input file name, and definitions for targets, features, and summary statistics, you can execute the data preparation workflow.

Load the configuration file and then call the ``create_training_dataset`` function:

.. code-block:: python

   import aiqclib as aq

   config_path = "~/aiqc_project/config/prepare_config.yaml"
   config = aq.read_config(config_path)
   aq.create_training_dataset(config)

Understanding the Output
------------------------
After the commands finishes, your main output directory (as defined by ``path_info_sets.common.base_path``, e.g., ``~/aiqc_project/data``) will contain a new folder named ``dataset_0001`` (derived from ``data_sets.dataset_folder_name``). Inside this folder, you will find several subdirectories, each representing a stage of the data preparation pipeline:

*   **summary**: Contains intermediate files with summary statistics of the input data, often used for normalization or feature scaling.
*   **select**: Stores data points identified as "good" (negative samples) and "bad" (positive samples) based on your target and QC flag definitions.
*   **locate**: Contains specific observation records for both positive and negative profiles, often after a proximity-based selection.
*   **extract**: Holds the features extracted from the observation records, ready for model consumption.
*   **training**: The final output directory. This contains the split training, validation, and test datasets in Parquet format, ready for model training and evaluation.

Next Steps
----------
Congratulations! You have successfully prepared your dataset, transforming raw data into a structured format with engineered features and appropriate splits. You are now ready to train your first machine learning model using ``aiqclib``.

Proceed to the next tutorial: :doc:`./training`.
