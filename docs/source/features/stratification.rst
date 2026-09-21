Stratification
===========================

The ``stratification`` feature is an observation-level feature describing the
vertical stability of the water column: whether the water below a level is
denser than the water above it, and by how much.

This is a different kind of feature from the anomaly scores. Those ask whether
a measurement disagrees with its neighbours, which is a statistical question.
This one asks whether the water column the measurement describes could exist,
which is a physical one. A column with lighter water underneath heavier water
overturns within minutes, so a profile that reports one is either catching a
rare, genuinely transient event or, far more often, reporting a bad
measurement.

Configuration: Setup
-------------------------------------

.. code-block:: yaml

   feature_sets:
     - name: feature_set_1
       features:
         - stratification

Configuration: Parameters
-------------------------------------

.. code-block:: yaml

   feature_param_sets:
     - name: feature_set_1_param_set_1
       params:
         - feature: stratification
           outputs: [ n2, n2_abs, unstable_flag ]
           params: { unstable_n2: 0.0 }
           stats_set: { type: raw }

================================ =========================================
Output                           What it holds
================================ =========================================
``sigma0_gradient``              The centred vertical gradient of the
                                 potential density anomaly, kg/m⁴,
                                 positive where density increases downward.
``n2``                           The Brunt-Vaisala frequency squared in
                                 1/s²: positive for a stable column,
                                 negative for an inverted one, near zero
                                 in a well mixed layer.
``n2_abs``                       Its magnitude, which says how strongly
                                 stratified a level is without saying in
                                 which direction.
``unstable_flag``                1 where ``n2`` is below ``unstable_n2``,
                                 else 0, and null where ``n2`` is null.
================================ =========================================

================================ ================== =====================
Parameter                        Default            Meaning
================================ ================== =====================
``unstable_n2``                  ``0.0``            The threshold below
                                                    which a level counts as
                                                    unstable. The default
                                                    means any inversion at
                                                    all.
``sigma0_column``                none               A density column
                                                    already in the input to
                                                    use instead of
                                                    recomputing one.
``min_depth_separation``         ``0.01``           Metres. A gradient
                                                    across neighbours
                                                    closer than this is
                                                    null rather than a
                                                    division by nearly
                                                    zero.
================================ ================== =====================

The input column parameters are the same as for
:doc:`derived_values`.

Relationship to the density inversion QC item
----------------------------------------------

``qc_density_inversion`` (RTQC14) tests the same physics and answers with a
flag: pass or fail against a tolerance. This feature hands the model the
quantity instead, so it can learn where the boundary lies rather than being
told. Both can be used together; they are computed from the same equation of
state.

Where the answer is null
-------------------------------------

The first and last level of every profile have no pair of neighbours to
difference across, so every output is null there. So is any level whose
temperature, salinity or pressure is missing or outside the domain of the
equation of state.
