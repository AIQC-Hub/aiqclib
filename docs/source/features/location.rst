Location
===========================

The ``location`` feature is a profile-level feature that represents the locations of sampling points. All observations belonging to the same profile generally have the same ``location`` feature values. Although ``aiqclib`` requires both ``latitude`` and ``longitude`` values, any other columns in the input dataset can be specified as the ``location`` feature.

Configuration: Setup
-------------------------------------

To include the ``location`` feature in your training and classification datasets, the value ``location`` needs to be specified in the ``feature_sets`` section.

.. code-block:: yaml

   feature_sets:
     - name: feature_set_1
       features:
         - location

Configuration: Parameters
-------------------------------------

The ``location`` feature requires two mandatory parameters: ``col_names`` and ``stats_set``.

*   The ``col_names`` parameter specifies the column names in the input dataset that will be used as the ``location`` feature.
*   The ``stats_set`` parameter specifies how the feature values are normalized. ``aiqclib`` currently supports ``raw`` and ``min_max`` as normalization methods. The ``name`` value in ``stats_set`` requires the same ``name`` in the ``feature_stats_sets`` section.

.. code-block:: yaml

   feature_param_sets:
     - name: feature_set_1_param_set_1
       params:
         - feature: location
           col_names: [ longitude, latitude ]
           stats_set: { type: min_max, name: location }

Configuration: Normalization
-------------------------------------

If the normalization method is not set to ``raw`` values, the summary statistics specified here will be used for normalization.

.. code-block:: yaml

   feature_stats_sets:
     - name: feature_set_1_stats_set_1
       min_max:
         - name: location
           stats: { longitude: { min: 14.5, max: 23.5 },
                    latitude: { min: 55, max: 66 } }

.. note::

   ``aiqclib`` offers helper functions to calculate the summary statistics (like min/max values). Please refer to the :doc:`../how-to/feature_normalization` guide for details.
