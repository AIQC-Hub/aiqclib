Geographic Context
===========================

The ``geo_context`` feature is an observation-level feature saying what kind of
place a profile was taken in, and where in the water column each level sits.

:doc:`location` already says where a profile was taken. What it cannot say is
what kind of place that is. The same temperature at 50 metres means one thing
over a 60 metre shelf, where the measurement is nearly on the bottom and the
water column is mixed by tides, and another over a 4000 metre basin, where it
is in the surface layer.

Configuration: Setup
-------------------------------------

.. code-block:: yaml

   feature_sets:
     - name: feature_set_1
       features:
         - geo_context

Configuration: Parameters
-------------------------------------

.. code-block:: yaml

   feature_param_sets:
     - name: feature_set_1_param_set_1
       params:
         - feature: geo_context
           outputs: [ bathymetry, normalized_depth, distance_to_bottom ]
           params: { bathymetry_column: bathymetry, positive_depth: true }
           stats_set: { type: raw }

================================ =========================================
Output                           What it holds
================================ =========================================
``bathymetry``                   The sea floor depth at the profile
                                 position, in metres positive downward
                                 whatever convention the input uses.
``coast_distance``               The distance to the nearest coast, passed
                                 through from its input column.
``normalized_depth``             The level's depth divided by the sea
                                 floor depth: 0 at the surface, 1 at the
                                 bottom, whatever the depth of the water.
``distance_to_bottom``           The metres of water beneath the level.
``deep_stable_layer``            1 where the level is deeper than
                                 ``deep_threshold`` and the water column
                                 there is stable but only weakly
                                 stratified.
================================ =========================================

================================ ================== =====================
Parameter                        Default            Meaning
================================ ================== =====================
``bathymetry_column``            ``bathymetry``     The sea floor depth
                                                    column.
``coast_distance_column``        ``coast_distance`` The coast distance
                                                    column.
``positive_depth``               ``true``           Whether larger values
                                                    in the bathymetry
                                                    column mean deeper.
``deep_threshold``               ``1000.0``         Metres below which a
                                                    level counts as deep.
``stability_threshold``          ``0.005``          kg/m⁴ density gradient
                                                    below which a stable
                                                    column counts as
                                                    weakly stratified.
``required``                     ``true``           Whether a missing
                                                    input column is an
                                                    error.
================================ ================== =====================

The grids are inputs, not lookups
-------------------------------------

``aiqclib`` does not read GEBCO or GSHHG. ``bathymetry`` and
``coast_distance`` are expected as columns of the input, exactly as the
:ref:`position_on_land <nrt-qc-items>` QC item expects the sea floor depth.
Sampling the grids is an upstream job, done where the input is assembled.

That keeps netCDF and shapefile libraries, their data files, and their
coordinate conventions out of this library. It also means the sampling is done
once, when the dataset is built, rather than on every run.

Both sign conventions are in use, which is why ``positive_depth`` exists.
Getting it wrong does not invert the feature: a sea floor at or above sea level
is not a sea floor, so the outputs are null rather than quietly wrong. Check
your data before enabling the feature.

Two ways of saying where in the water column
---------------------------------------------

``normalized_depth`` and ``distance_to_bottom`` are not redundant.
``normalized_depth`` makes a shelf cast and a deep cast comparable: half way
down is 0.5 in both. ``distance_to_bottom`` says something
``normalized_depth`` cannot: ten metres off the bottom is ten metres off the
bottom in either place, and that is where bottom effects appear.

When the dataset has no bathymetry
-------------------------------------

A configured column that is missing is an error, as it is for the QC item.
Setting ``required`` to ``false`` emits the columns that depend on it as null
instead, with a warning.

That is worth having when one configuration is run over several regions and
only some of the inputs carry the grids: the training frame keeps the same
schema either way, so the model's feature list does not change from region to
region. Use it deliberately, since a column of nulls looks like missing data
rather than a column nobody computed.
