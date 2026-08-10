Configuration of Near-Real Time QC
========================================
The ``nrt_qc`` workflow (``stage="nrt_qc"``) applies automated real-time QC
tests to temperature and salinity profile data and writes the input parquet
enriched with one flag column per QC item plus a final NRT flag per variable
(see the :doc:`../how-to/nrt_qc` guide for the workflow overview).

Prepare **one configuration file per region** (e.g. Arctic, Baltic,
Mediterranean): the files share the same structure and differ only in
region-dependent parameters such as the regional ranges.

Detailed Configuration Sections
-------------------------------

`path_info_sets`
^^^^^^^^^^^^^^^^
Defines the location of the input data and the output directories.

*   **common.base_path**: The root output directory.
*   **input.base_path**: The directory containing the input parquet files.
*   **concat.step_folder_name**: The subdirectory for the final output
    parquet (the intermediate flag file of the ``qc`` step and the
    comparison reports of the ``compare`` step default to folders named
    after their steps).

.. code-block:: yaml

   path_info_sets:
     - name: nrt_qc_path_1
       common:
         base_path: /path/to/data
       input:
         base_path: /path/to/input
         step_folder_name: ""
       concat:
         step_folder_name: nrt_qc

.. note::
   A leading ``~`` in any ``base_path`` is expanded to your home directory, so
   ``~/aiqc_project/data`` means the same directory it would in a shell.

`qc_variable_sets`
^^^^^^^^^^^^^^^^^^
The variables to run the QC items on. Only ``name`` is required. The
optional ``flag`` names an **existing** NRT QC flag column in the input and
enables the flag comparison report for that variable;
``pos_flag_values`` / ``neg_flag_values`` additionally enable the binary
agreement metrics (same convention as the other modules).

.. code-block:: yaml

   qc_variable_sets:
     - name: qc_variable_set_1
       variables:
         - name: temp
           flag: temp_qc # Optional: existing NRT QC flag column
           pos_flag_values: [ 4, 6, 7 ] # Optional: for agreement metrics
           neg_flag_values: [ 1 ]
         - name: psal
           flag: psal_qc
           pos_flag_values: [ 4, 6, 7 ]
           neg_flag_values: [ 1 ]

.. note::
   The flag column may hold integers, strings or floats, and the values may be
   written as ``4`` or as ``"4"``; both sides are read as whole numbers. See
   :ref:`qc-flag-columns` for the full table.

`qc_item_sets`
^^^^^^^^^^^^^^
The QC items to apply. An item runs only if listed; ``params`` overrides the
built-in defaults per key, and the optional ``fail_flag`` (3 or 4, default 4)
softens a failing test to "probably bad".

For what each item checks, rather than how it is configured, see
:ref:`nrt-qc-items` in the how-to guide. The same items can be used as model
input features — see :doc:`../how-to/qc_items_as_features`.

================================ ======= ============ =========================================
Item                             RTQC    Level        Output column(s)
================================ ======= ============ =========================================
``impossible_date``              RTQC2   profile      ``qc_impossible_date``
``impossible_location``          RTQC3   profile      ``qc_impossible_location``
``global_range``                 RTQC6   observation  ``{var}_qc_global_range``
``regional_range``               RTQC7   observation  ``{var}_qc_regional_range``
``pressure_increasing``          RTQC8   observation  ``qc_pressure_increasing``
``spike``                        RTQC9   observation  ``{var}_qc_spike``
``gradient``                     RTQC11  observation  ``{var}_qc_gradient``
``digit_rollover``               RTQC12  observation  ``{var}_qc_digit_rollover``
``stuck_value``                  RTQC13  profile      ``{var}_qc_stuck_value``
``density_inversion``            RTQC14  observation  ``temp_qc_...`` and ``psal_qc_...``
``temp_to_psal``                 —       observation  ``psal_qc_temp_to_psal``
================================ ======= ============ =========================================

.. code-block:: yaml

   qc_item_sets:
     - name: qc_item_set_1
       items:
         - name: impossible_date
         - name: impossible_location
         - name: global_range
           params: { temp: { min: -2.5, max: 40.0 },
                     psal: { min: 2.0, max: 41.0 } }
         - name: regional_range # EDIT: ranges of this file's region
           params: { temp: { min: 10.0, max: 40.0 },
                     psal: { min: 2.0, max: 40.0 } }
         - name: pressure_increasing
         - name: spike
           params: { temp: { shallow: 6.0, deep: 2.0 },
                     psal: { shallow: 0.9, deep: 0.3 },
                     depth_threshold: 500 }
         - name: gradient
           params: { temp: { shallow: 9.0, deep: 3.0 },
                     psal: { shallow: 1.5, deep: 0.5 },
                     depth_threshold: 500 }
         - name: digit_rollover
           params: { temp: 10.0, psal: 5.0 }
         - name: stuck_value
         - name: density_inversion
           params: { threshold: 0.03 }
         - name: temp_to_psal

Notes:

*   **regional_range** has no built-in defaults — supply your region's
    ranges, or the item raises an error (no silent pass).
*   **spike** / **gradient** use the ``shallow`` threshold below
    ``depth_threshold`` (in decibars) and ``deep`` at or beyond it.
*   **density_inversion** computes the potential density anomaly sigma-0
    (UNESCO 1983) and allows inversions up to ``threshold`` (kg/m³); it
    flags temperature and salinity jointly.
*   **temp_to_psal** propagates the final temperature flag onto salinity
    (see the :doc:`../how-to/nrt_qc` guide); omit it for independently
    measured salinity.

`step_class_sets`
^^^^^^^^^^^^^^^^^
The Python classes for the four NRT QC steps: reading the input (``input``),
running the QC items (``qc``), aggregating the final flags and writing the
output (``concat``), and the optional flag comparison (``compare``).

.. code-block:: yaml

   step_class_sets:
     - name: nrt_qc_step_set_1
       steps:
         input: InputDataSetAll
         qc: QCDataSetAll
         concat: ConcatDataSetAll
         compare: CompareFlagsAll

`step_param_sets`
^^^^^^^^^^^^^^^^^
Optional parameters per step. The ``input`` step shares the sub-step options
of the other modules (column renaming, row filtering, column creation and
validation); the NRT QC template disables row filtering by default.

.. code-block:: yaml

   step_param_sets:
     - name: nrt_qc_param_set_1
       steps:
         input: { sub_steps: { rename_columns: false,
                               filter_rows: false } }
         qc: { }
         concat: { }
         compare: { }

`nrt_qc_sets`
^^^^^^^^^^^^^
Ties everything together for one NRT QC run: the dataset folder, the input
file, and the named sub-configurations to use.

.. code-block:: yaml

   nrt_qc_sets:
     - name: nrt_qc_0001
       dataset_folder_name: nrt_qc_0001
       input_file_name: nrt_cora_bo_4.parquet
       path_info: nrt_qc_path_1
       qc_variable_set: qc_variable_set_1
       qc_item_set: qc_item_set_1
       step_class_set: nrt_qc_step_set_1
       step_param_set: nrt_qc_param_set_1
