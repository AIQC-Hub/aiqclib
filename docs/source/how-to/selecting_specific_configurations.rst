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

Generalizing to Other Configuration Types and Stages
------------------------------------------------------

This same approach applies to selecting specific configurations for other stages of your machine learning workflow. If your configuration file defines multiple named entries within sections like ``data_sets`` (for the *Dataset Preparation* stage), ``training_sets`` (for the *Training & Evaluation* stage), or ``classification_sets`` (for the *Classification* stage), you can use the ``set_name`` parameter with ``read_config`` to load the desired one. The ``set_name`` parameter will expect the specific ``name`` property of the entry you wish to select from the respective section in your configuration file.