Basic Values
===========================

The ``basic_values`` feature is an observation-level feature that represents the actual observation values, such as temperature and salinity. Any columns in the input dataset can be specified as the ``basic_values`` feature.

Configuration: Setup
-------------------------------------

To include the ``basic_values`` feature in your training and classification datasets, the value ``basic_values`` needs to be specified in the ``feature_sets`` section.

.. code-block:: yaml

   feature_sets:
     - name: feature_set_1
       features:
         - basic_values

Configuration: Parameters
-------------------------------------

The ``basic_values`` feature requires two mandatory parameters: ``col_names`` and ``stats_set``.

*   The ``col_names`` parameter specifies the column names in the input dataset that will be used as the ``basic_values`` feature.
*   The ``stats_set`` parameter specifies how the feature values are normalized. ``aiqclib`` currently supports ``raw`` and ``min_max`` as normalization methods. The ``name`` value in ``stats_set`` must correspond to a ``name`` in the ``feature_stats_sets`` section.

.. code-block:: yaml

   feature_param_sets:
     - name: feature_set_1_param_set_1
       params:
         - feature: basic_values
           col_names: [ temp, psal, pres ]
           stats_set: { type: min_max, name: basic_values3 }

Configuration: Normalization
-------------------------------------

If the normalization method is not set to ``raw``, the summary statistics specified here will be used for normalization.

.. code-block:: yaml

   feature_stats_sets:
     - name: feature_set_1_stats_set_1
       min_max:
         - name: basic_values3
           stats: { temp: { min: 0, max: 20 },
                    psal: { min: 0, max: 20 },
                    pres: { min: 0, max: 200 } }

.. note::

   ``aiqclib`` offers helper functions to calculate summary statistics (like min/max values). Please refer to the :doc:`../how-to/feature_normalization` guide for details.
