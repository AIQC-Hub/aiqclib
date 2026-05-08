Step 2: Dataset Preparation
===========================

The dataset preparation workflow is the first crucial step in the ``aiqclib`` pipeline. It's designed to prepare feature data sets from your raw data. This includes creating training, validation, and test data sets.

This entire process is driven by a YAML configuration file, ensuring your data preparation is repeatable, transparent, and easy to manage across different experiments.

.. admonition:: A Note on Running the Examples

   The examples in these tutorials are presented as commands suitable for an interactive Python session (e.g., in a terminal with ``python`` or ``ipython``, or within a Jupyter Notebook/Lab).

   However, you are encouraged to use the method you are most comfortable with. The code can be run in several ways:

   *   **In an Interactive Python Session:** Launch Python (``python``) or IPython (``ipython``) and paste the code line by line. This is great for quick tests and exploration.
   *   **As Python Scripts:** Copy the code into a ``.py`` file (e.g., ``prepare_data.py``) and execute it from your terminal with ``python your_script_name.py``. This is suitable for automation and batch processing.
   *   **In a Jupyter Notebook or Lab:** This is a fantastic option for experimentation, as it allows you to run code in cells, add notes, and visualize results interactively.

   Feel free to adapt the examples to your preferred environment.

Getting the Example Data
------------------------

This tutorial uses the Copernicus Marine NRT CTD dataset, publicly available on ``Kaggle``. Before proceeding, let's set up your project directory structure and download the necessary data.

First, create the directories for your project. This structure will be used consistently throughout the tutorials:

.. code-block:: bash

   # Create a main project directory for all aiqclib outputs and configs
   mkdir -p ~/aiqc_project

   # Create subdirectories for configuration files and raw input data
   mkdir -p ~/aiqc_project/config
   mkdir -p ~/aiqc_project/input

Now, choose one of the following options to download the dataset. Both methods will place the required data in ``~/aiqc_project/input/``.

.. tabs::

   .. tab:: Option 1: Kaggle API (Recommended)

      This method is ideal for reproducibility and for users who frequently work with Kaggle datasets.

      1. **Install and configure the Kaggle API:**
         If you haven't already, install the ``kaggle`` client and set up your API credentials.

         .. code-block:: bash

            pip install kaggle

         Follow the official `Kaggle API authentication instructions <https://www.kaggle.com/docs/api#getting-started-installation-&-authentication>`_ to obtain your ``kaggle.json`` file and place it in the correct location (``~/.kaggle/``).

      2. **Download and unzip the data:**
         This single command downloads and extracts the dataset directly into your ``input`` folder.

         .. code-block:: bash

            kaggle datasets download -d takaya88/copernicus-marine-nrt-ctd-data-for-aiqc -p ~/aiqc_project/input --unzip

   .. tab:: Option 2: cURL (Quickstart)

      This method is the fastest way to get the data, as it requires no extra tools or setup beyond standard command-line utilities.

      1. **Download the zip file using cURL:**

         .. code-block:: bash

            curl -L -o ~/aiqc_project/input/data.zip \
              https://www.kaggle.com/api/v1/datasets/download/takaya88/copernicus-marine-nrt-ctd-data-for-aiqc

      2. **Unzip the file:**
         Extract the downloaded archive into your input directory.

         .. code-block:: bash

            unzip ~/aiqc_project/input/data.zip -d ~/aiqc_project/input/

----------

After following either set of instructions, you should now have a file named ``nrt_cora_bo_4.parquet`` inside ``~/aiqc_project/input/``.

Required Input Data Structure
-----------------------------
``aiqclib`` expects your raw input data (a Parquet file) to contain specific columns, which are crucial for identifying unique profiles and observations. If your raw data already contains these, you're good to go. Otherwise, you may need to preprocess your data to create them.

The required columns are:

*   **platform_code**: A unique identifier for the measurement platform (e.g., buoy, ship).
*   **profile_no**: A unique, sequential number identifying each distinct "profile" (a set of measurements taken at a specific time and location) within a ``platform_code``.
*   **profile_timestamp**: The exact datetime of the profile. This column should be of a datetime type (e.g., Pandas/Polars datetime, or similar).
*   **longitude**: The longitude of the measurement profile.
*   **latitude**: The latitude of the measurement profile.
*   **observation_no**: A unique, sequential number identifying each individual observation (row) within a ``profile_no``.
*   **pres**: Pressure values for each observation.

.. important::

   If your raw data lacks ``profile_no``, ``profile_timestamp``, or ``observation_no``, you will need to generate them. For detailed examples and helper code on how to perform these common data preprocessing steps (e.g., converting float timestamps, generating unique IDs), please refer to the :doc:`../../how-to/data_preprocessing_utilities` guide.

The Dataset Preparation Workflow
--------------------------------
The ``aiqclib`` data preparation workflow consists of three main programmatic steps: generating a configuration template, customizing this template to match your data and desired processing, and finally running the preparation script.

Step 2.1: Generate the Configuration Template
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
First, use ``aiqclib`` to generate a boilerplate configuration template. This file will contain all the necessary sections for the data preparation task, which you will then customize.

.. code-block:: python

   import aiqclib as aq
   import os

   config_path = os.path.expanduser("~/aiqc_project/config/prepare_config.yaml")
   aq.write_config_template(
       file_name=config_path,
       stage="prepare"
   )

Step 2.2: Customize the Configuration File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Now, open the newly created ``~/aiqc_project/config/prepare_config.yaml`` in a text editor. You need to tell ``aiqclib`` where to find your input data, where to save the processed output, and define your targets and features.

You will primarily focus on updating the following sections:

*   **path_info_sets**: Define your input and output directories.
*   **target_sets**: Specify your prediction targets and their quality control flags.
*   **summary_stats_sets**: Provide settings for summary statistics.
*   **feature_sets & feature_param_sets**: List the feature engineering methods and their parameters.
*   **feature_stats_sets**: Provide statistics for feature normalization.
*   **data_sets**: Assemble the full pipeline by linking the named blocks.

**Updating path_info_sets and data_sets:**
Update your ``prepare_config.yaml`` to match the following for the ``path_info_sets`` and ``data_sets`` sections, replacing the placeholder paths with the ones you created in "Getting the Example Data".

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
   The ``prepare_config.yaml`` can be quite detailed. For a complete reference of all available configuration options, please consult the dedicated :doc:`../../configuration/preparation` page.

.. note::

   ``dmaclib`` provides methods to down-sample the negative data set. Please refer to the :doc:`../../how-to/down_sampling_negative` guide for details.

Step 2.2: Run the Preparation Process
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Once you have customized your ``prepare_config.yaml`` with the correct paths, input file name, and definitions for targets, features, and summary statistics, you can execute the data preparation workflow.

Load the configuration file and then call the ``create_training_dataset`` function:

.. code-block:: python

   import aiqclib as aq
   import os

   config_path = os.path.expanduser("~/aiqc_project/config/prepare_config.yaml")
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
