Neighboring Values (Up and Down)
==========================================

The ``flank_up`` and ``flank_down`` features are observation-level features that represent the neighboring values of an observation, such as temperature and salinity. Although any columns in the input dataset can be specified for these features, they are usually coupled with the variables used in the ``basic_values`` feature.

Configuration: Setup
-------------------------------------

To include the ``flank_up`` and/or ``flank_down`` features in your training and classification datasets, the values ``flank_up`` and/or ``flank_down`` need to be specified in the ``feature_sets`` section.

.. code-block:: yaml

   feature_sets:
     - name: feature_set_1
       features:
         - flank_up
         - flank_down

Configuration: Parameters
-------------------------------------

Both ``flank_up`` and ``flank_down`` features require two common mandatory parameters (``col_names`` and ``stats_set``) and one feature-specific parameter (``flank_up`` or ``flank_down``).

*   The ``col_names`` parameter specifies the column names in the input dataset that will be used for the ``flank_up`` and ``flank_down`` features.
*   The ``stats_set`` parameter specifies how the feature values are normalized. ``aiqclib`` currently supports ``raw`` and ``min_max`` as normalization methods. The ``name`` value in ``stats_set`` must correspond to a ``name`` in the ``feature_stats_sets`` section.
*   The ``flank_up`` and ``flank_down`` parameters specify the number of neighboring values to include in the feature.

.. code-block:: yaml

   feature_param_sets:
     - name: feature_set_1_param_set_1
       params:
         - feature: flank_up
           col_names: [temp, psal, pres]
           stats_set: { type: min_max, name: basic_values3 }
           flank_up: 5
         - feature: flank_down
           col_names: [temp, psal, pres]
           stats_set: { type: min_max, name: basic_values3 }
           flank_down: 5

Configuration: Normalization
-------------------------------------

If the normalization method is not set to ``raw``, the summary statistics specified here will be used for normalization. These features normally use the same summary statistics as the corresponding ``basic_values`` feature.

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
