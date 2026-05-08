Profile Summary Statistics
=======================================

The ``profile_summary_stats`` feature is a profile-level feature that represents the summary statistics of specified variables. All observations belonging to the same profile generally have the same ``profile_summary_stats`` feature values. The ``profile_summary_stats`` feature can contain the following nine statistics:

1.  **min**: minimum
2.  **max**: maximum
3.  **mean**: mean
4.  **median**: median
5.  **pct25**: 25th percentile
6.  **pct75**: 75th percentile
7.  **pct2.5**: 2.5th percentile
8.  **pct97.5**: 97.5th percentile
9.  **sd**: standard deviation

Configuration: Summary Statistics
-------------------------------------

The ``profile_summary_stats`` feature requires the calculation of summary statistics prior to feature extraction. This can be specified in the ``summary_stats_sets`` section of a configuration file. The variables used for the feature should be specified in ``col_names``.

.. code-block:: yaml

   summary_stats_sets:
     - name: summary_stats_set_1
       stats:
         - name: profile_summary_stats
           col_names: [ temp, psal, pres ]

Configuration: Setup
-------------------------------------

To include the ``profile_summary_stats`` feature in your training and classification datasets, the value ``profile_summary_stats`` needs to be specified in the ``feature_sets`` section.

.. code-block:: yaml

   feature_sets:
     - name: feature_set_1
       features:
         - profile_summary_stats

Configuration: Parameters
-------------------------------------

The ``profile_summary_stats`` feature requires three mandatory parameters: ``col_names``, ``summary_stats_names``, and ``stats_set``.

*   The ``col_names`` parameter specifies the column names in the input dataset that will be used for the ``profile_summary_stats`` feature.
*   The ``summary_stats_names`` parameter specifies the names of the summary statistics to be used as features.
*   The ``stats_set`` parameter specifies how the feature values are normalized. ``aiqclib`` currently supports ``raw`` and ``min_max`` as normalization methods. The ``name`` value in ``stats_set`` must correspond to a ``name`` in the ``feature_stats_sets`` section.

.. code-block:: yaml

   feature_param_sets:
     - name: feature_set_1_param_set_1
       params:
         - feature: profile_summary_stats
           col_names: [ temp, psal, pres ]
           summary_stats_names: [ mean, median, sd, pct25, pct75 ]
           stats_set: { type: min_max, name: profile_summary_stats }

Configuration: Normalization
-------------------------------------

If the normalization method is not set to ``raw``, the summary statistics specified here will be used for normalization.

.. code-block:: yaml

   feature_stats_sets:
     - name: feature_set_1_stats_set_1
       min_max:
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

.. note::

   ``aiqclib`` offers helper functions to calculate summary statistics (like min/max values). Please refer to the :doc:`../how-to/feature_normalization` guide for details.
