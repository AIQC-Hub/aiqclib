Feature Normalization
=====================

``aiqclib`` uses ``XGBoost`` by default, which does not need normalized
features. Non-tree-based models such as ``SVM`` do, so each feature can be
normalized independently by setting ``stats_set.type`` in
``feature_param_sets``.

Normalization methods
---------------------

.. list-table::
   :header-rows: 1
   :widths: 24 32 44

   * - Type
     - Formula
     - Values come from
   * - ``raw`` (default)
     - no scaling
     - —
   * - ``min_max``
     - ``(x - min) / (max - min)``
     - values you enter in ``feature_stats_sets``
   * - ``auto_min_max``
     - ``(x - min) / (max - min)``
     - derived from your data automatically
   * - ``standard``
     - ``(x - mean) / sd``
     - derived from your data automatically

Templates from ``write_config_template`` use ``raw`` for every feature.
``day_of_year`` is always converted with ``convert: cosine`` and is not
affected by ``stats_set``.

Automatic normalization
-----------------------

For ``auto_min_max`` and ``standard`` you do not enter any values. Set the
type on each feature and leave ``feature_stats_sets`` as a name-only entry:

.. code-block:: yaml

   feature_param_sets:
     - name: feature_set_1_param_set_1
       params:
         - feature: basic_values
           stats_set: { type: standard, name: basic_values }
           col_names: [ temp, psal, pres ]
         - feature: profile_summary_stats
           stats_set: { type: auto_min_max, name: profile_summary_stats }
           col_names: [ temp, psal, pres ]
           summary_stats_names: [ mean, median, sd, pct25, pct75 ]

   feature_stats_sets:
     - name: feature_set_1_stats_set_1   # values are filled in automatically

During ``prepare`` the values are computed from the data and written to a
normalization file. During ``classify`` that file is read back, so the same
scaling is reapplied without re-entering values or needing the training data.

The file defaults to ``normalization_stats.yaml`` in a ``normalize`` folder
under the dataset. To change the name, set it on a ``normalize`` step; its
location can be set under ``path_info_sets`` like any other step:

.. code-block:: yaml

   step_param_sets:
     - name: step_param_set_1
       steps:
         normalize: { file_name: normalization_stats.yaml }

Point the ``classify`` configuration at the same file produced during
``prepare``.

Manual normalization (``min_max``)
----------------------------------

Use ``min_max`` to set explicit ranges yourself.

**1. Generate a template with worked examples.** The ``full`` template
contains a complete ``min_max`` example for every feature.

.. code-block:: python

   import aiqclib as aq

   aq.write_config_template(file_name="prepare_config.yaml",
                            stage="prepare", extension="full")
   aq.write_config_template(file_name="classification_config.yaml",
                            stage="classify", extension="full")

**2. Inspect your data to choose ranges.**

.. code-block:: python

   import aiqclib as aq

   input_file = "~/aiqc_project/input/nrt_cora_bo_4.parquet"
   print(aq.format_summary_stats(aq.get_summary_stats(input_file, "all")))
   print(aq.format_summary_stats(aq.get_summary_stats(input_file, "profiles")))

**3. Set the type/name and the values.** In ``feature_param_sets`` the
``name`` links each feature to a block in ``feature_stats_sets``:

.. code-block:: yaml

   feature_param_sets:
     - name: feature_set_1_param_set_1
       params:
         - feature: location
           stats_set: { type: min_max, name: location }
           col_names: [ longitude, latitude ]
         - feature: basic_values
           stats_set: { type: min_max, name: basic_values3 }
           col_names: [ temp, psal, pres ]

.. code-block:: yaml

   feature_stats_sets:
     - name: feature_set_1_stats_set_1
       min_max:
         - name: location
           stats: { longitude: { min: 14.5, max: 23.5 },
                    latitude: { min: 55, max: 66 } }
         - name: basic_values3
           stats: { temp: { min: 0, max: 20 },
                    psal: { min: 0, max: 20 },
                    pres: { min: 0, max: 200 } }
         # profile_summary_stats uses a nested {variable: {stat: {min, max}}}
         # form; see the full template for the complete example.

Unlike the automatic methods, ``min_max`` values are read from each
configuration directly (no file is written), so set the same values in both
the ``prepare`` and ``classify`` configurations.