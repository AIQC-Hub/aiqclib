Day of the Year
=================================

The ``day_of_year`` feature is a profile-level feature that represents the day of the year on which the profile was sampled. All observations belonging to the same profile generally have the same ``day_of_year`` feature value. In the current version of the library, only the ``profile_timestamp`` data can be used to produce this feature.

Configuration: Setup
-------------------------------------

To include the ``day_of_year`` feature in your training and classification datasets, the value ``day_of_year`` needs to be specified in the ``feature_sets`` section.

.. code-block:: yaml

   feature_sets:
     - name: feature_set_1
       features:
         - day_of_year

Configuration: Parameters
-------------------------------------

The ``day_of_year`` feature accepts two parameters: ``col_names`` and ``convert``.

*   The ``col_names`` parameter should be ``[ profile_timestamp ]`` in the current version.
*   The ``convert`` parameter can be set to ``sine``, ``cosine``, or a non-``sine`` value. If ``sine`` or ``cosine`` is specified, a sine or cosine conversion is performed; otherwise, the raw values are used.

.. code-block:: yaml

   feature_param_sets:
     - name: feature_set_1_param_set_1
       params:
         - feature: day_of_year
           col_names: [ profile_timestamp ]
           convert: cosine

Configuration: Normalization
-------------------------------------

Normalization methods are not available for this feature.
