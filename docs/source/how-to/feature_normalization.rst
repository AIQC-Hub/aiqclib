Feature Normalization
===========================

This guide provides practical examples of how to normalize feature values by specifying the required entries in configuration files. Although ``aiqclib`` uses ``XGBoost`` by default, which does not require normalized feature values, other non–tree-based machine learning methods, such as ``SVM``, do require feature normalization for their input data.

Required Steps
---------------------------

1. Generate configuration template
2. Calculate summary statistics
3. Set entries in the configuration files

Generate the Configuration Template
-------------------------------------

First, use ``aiqclib`` to generate the boilerplate configuration templates specifically for the ``prepare`` and ``classify`` workflows.

.. code-block:: python

   import aiqclib as aq
   import os

   # Define the path for the config file
   config_file_prepare = os.path.expanduser("~/aiqc_project/config/prepare_config.yaml")
   config_file_classify = os.path.expanduser("~/aiqc_project/config/classification_config.yaml")

   # This creates 'prepare_config.yaml' and 'classification_config.yaml'
   aq.write_config_template(
       file_name=config_file_prepare,
       stage="prepare",
       extension="full"
   )
   aq.write_config_template(
       file_name=config_file_classify,
       stage="classify",
       extension="full"
   )

Calculate Summary Statistics
-------------------------------------
The following Python commands, utilizing ``aiqclib``, can provide all necessary information to update the values in ``summary_stats_sets`` based on your actual data:

.. code-block:: python

   import aiqclib as aq

   input_file = "~/aiqc_project/input/nrt_cora_bo_4.parquet"

   stats_all = aq.get_summary_stats(input_file, "all")
   print(aq.format_summary_stats(stats_all))

   stats_profiles = aq.get_summary_stats(input_file, "profiles")
   print(aq.format_summary_stats(stats_profiles))

Set Entries in the Configuration Files
---------------------------------------
Entries in the ``feature_param_sets`` and ``feature_stats_sets`` sections in both ``prepare_config.yaml`` and ``classification_config.yaml`` need to be updated.

`feature_param_sets`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*   **params.stats_set.type**: ``aiqclib`` currently provides only the ``min_max`` normalization.
*   **params.stats_set.name**: The name of the normalization values that should be matched with the entry in ``feature_param_sets``.

.. code-block:: yaml
   :emphasize-lines: 5, 11, 15, 19, 23

   feature_param_sets:
     - name: feature_set_1_param_set_1
       params:
         - feature: location
           stats_set: { type: min_max, name: location }
           col_names: [ longitude, latitude ]
         - feature: day_of_year
           convert: cosine
           col_names: [ profile_timestamp ]
         - feature: profile_summary_stats
           stats_set: { type: min_max,  name: profile_summary_stats }
           col_names: [ temp, psal, pres ]
           summary_stats_names: [ mean, median, sd, pct25, pct75 ]
         - feature: basic_values
           stats_set: { type: min_max, name: basic_values3 }
           col_names: [ temp, psal, pres ]
         - feature: flank_up
           flank_up: 5
           stats_set: { type: min_max, name: basic_values3 }
           col_names: [ temp, psal, pres ]
         - feature: flank_down
           flank_down: 5
           stats_set: { type: min_max, name: basic_values3 }
           col_names: [ temp, psal, pres ]

`feature_stats_sets`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You need to update the stats values in the configuration files based on the results from ``aq.get_summary_stats`` and ``aq.format_summary_stats``.

.. code-block:: yaml

   feature_stats_sets:
     - name: feature_set_1_stats_set_1
       min_max:
         - name: location
           stats: { longitude: { min: 14.5, max: 23.5 },
                    latitude: { min: 55, max: 66 } }
         - name: profile_summary_stats
           stats: { temp: { mean: { min: 0, max: 12.5 },
                            median: { min: 0, max: 15 },
                            sd: { min: 0, max: 6.5 },
                            pct25: { min: 0, max: 12 },
                            pct75: { min: 1, max: 19 } },
                    psal: { mean: { min: 2.9, max: 12 },
                            median: { min: 2.9, max: 12 },
                            sd: { min: 0, max: 4 },
                            pct25: { min: 2.5, max: 8.5 },
                            pct75: { min: 3, max: 16 } },
                    pres: { mean: { min: 24, max: 105 },
                            median: { min: 24, max: 105 },
                            sd: { min: 13, max: 60 },
                            pct25: { min: 12, max: 53 },
                            pct75: { min: 35, max: 156 } } }
         - name: basic_values3
           stats: { temp: { min: 0, max: 20 },
                    psal: { min: 0, max: 20 },
                    pres: { min: 0, max: 200 } }
