Selecting Specific Configurations
=====================================

This guide demonstrates how to select a specific configuration (e.g., for a dataset, training set, or classification set) when multiple options are defined within a single configuration file. The ``read_config`` function in ``aiqclib`` allows you to easily specify which named configuration to load using the ``set_name`` parameter. This parameter selects an entry from ``data_sets`` in the *Dataset Preparation* stage, ``training_sets`` in the *Training & Evaluation* stage, or ``classification_sets`` in the *Classification* stage.

Example: Selecting a Data Set
-------------------------------

Consider a ``prepare_config.yaml`` file that defines multiple ``data_sets``, such as ``dataset_0001`` and ``dataset_0002``:

.. code-block:: yaml

   data_sets:
     - name: dataset_0001
       dataset_folder_name: dataset_0001
       input_file_name: nrt_cora_bo_4.parquet
       path_info: data_set_1
       target_set: target_set_1
       # ... other set references would follow here
     - name: dataset_0002
       dataset_folder_name: dataset_0002
       input_file_name: nrt_cora_bo_5.parquet
       path_info: data_set_1
       target_set: target_set_1
       # ... other set references would follow here

To use a specific data set from these defined options for your data preparation stage, pass its ``name`` to the ``set_name`` parameter of the ``aq.read_config`` function.

For example, to select ``dataset_0002``, you would use:

.. code-block:: python

   import aiqclib as aq

   config_path = "~/aiqc_project/config/prepare_config.yaml"
   config = aq.read_config(config_path, set_name="dataset_0002")

This ``config`` object will now contain the parameters for ``dataset_0002``, ready for further processes.

Checking What a Configuration Resolves To
-------------------------------------------

Printing a configuration object summarizes what it resolved to, which is the
quickest way to confirm you selected the entry you meant:

.. code-block:: python

   config = aq.read_config(config_path, set_name="dataset_0001")
   print(config)

.. code-block:: text

   DataSetConfig: dataset_0001
     source    prepare_config.yaml
     section   data_sets (2 entries)
     schema    valid
     targets   temp (flag temp_qc, pos [4, 6, 7], neg [1])
               psal (flag psal_qc, pos [4, 6, 7], neg [1])
     features  location, day_of_year, profile_summary_stats, basic_values, flank_up,
               flank_down
     input     /path/to/input/nrt_cora_bo_4.parquet
     filters   remove_years [2023]
     steps     input    InputDataSetA    /path/to/input
               summary  SummaryDataSetA  /path/to/data/dataset_0001/summary
               select   SelectDataSetA   /path/to/data/dataset_0001/select
               locate   LocateDataSetA   /path/to/data/dataset_0001/locate
               extract  ExtractDataSetA  /path/to/data/dataset_0001/extract
               split    SplitDataSetA    /path/to/data/dataset_0001/training

The ``steps`` block is the useful part: it names the class that runs each step
and the directory its output lands in, with ``~`` and the dataset folder
already resolved. Reading it before a long run is cheaper than discovering
afterwards that everything was written somewhere unexpected.

The other rows are worth a glance for the mistakes they make visible:

*   ``targets`` — the variables a model is built for, each with the QC flag
    column it labels from and the flag values counted as positive (bad) and
    negative (good). See :ref:`choosing-targets`.
*   ``filters`` — the active ``keep_years`` / ``remove_years`` row filters. A
    filter naming years your input does not cover empties the dataset, which
    is otherwise only reported once the pipeline reaches the step that cannot
    proceed without rows.
*   ``schema`` — whether the file validates against the stage's schema, and
    the first error if it does not. This is reported before an entry is
    selected too, so printing a configuration that will not load still tells
    you why.

A configuration that has not selected an entry yet lists the names it offers
instead:

.. code-block:: python

   config = aq.read_config(config_path, auto_select=False)
   print(config)

.. code-block:: text

   DataSetConfig: <nothing selected>
     source    prepare_config.yaml
     section   data_sets (2 entries)
     schema    valid
     entries   dataset_0001, dataset_0002
     (call select(<name>) to resolve one of the entries above)

The same text is available as a string from ``config.summary()``, for logging
it alongside a run rather than printing it.

Inspecting a Stage's Defaults
--------------------------------

``read_config_template`` takes the same ``stage`` and ``extension`` arguments
as ``write_config_template`` but returns the configuration object rather than
writing the YAML to a file, which is the shortest way to look at what a stage
starts from:

.. code-block:: python

   print(aq.read_config_template(stage="prepare"))

The returned object can also be customized in code — adjusting ``config.data``
— and passed straight to a workflow, so a configuration need not exist on disk
at all. A template carries placeholder paths, so set ``path_info`` and
``input_file_name`` on it before running anything with it:

.. code-block:: python

   config = aq.read_config_template(stage="prepare")
   config.data["path_info"]["common"]["base_path"] = "~/aiqc_project/data"
   config.data["path_info"]["input"]["base_path"] = "~/aiqc_project/input"
   config.data["input_file_name"] = "my_profiles.parquet"

   print(config)   # confirm the paths and filters before the run
   aq.create_training_dataset(config)

.. note::

   The prepare template ships with ``remove_years: [2023]`` as an example row
   filter. It shows up in the ``filters`` row of the summary above — clear it
   in ``step_param_set`` unless you want that year dropped.

Generalizing to Other Configuration Types and Stages
------------------------------------------------------

This same approach applies to selecting specific configurations for other stages of your machine learning workflow. If your configuration file defines multiple named entries within sections like ``data_sets`` (for the *Dataset Preparation* stage), ``training_sets`` (for the *Training & Evaluation* stage), or ``classification_sets`` (for the *Classification* stage), you can use the ``set_name`` parameter with ``read_config`` to load the desired one. The ``set_name`` parameter will expect the specific ``name`` property of the entry you wish to select from the respective section in your configuration file.