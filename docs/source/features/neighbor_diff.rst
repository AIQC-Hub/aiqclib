Neighbour Differences
===========================

The ``neighbor_diff`` feature is an observation-level feature holding the
difference between a measurement and the measurements a given number of levels
above or below it.

:doc:`neigbouring_values` (``flank_up`` and ``flank_down``) already hand the
model those neighbouring *values*. Differencing them first does two things.
It removes the baseline, so the same 0.5 degree step means the same thing in
the Baltic and the Mediterranean rather than being a different pair of numbers
in each. And it saves the model from having to learn the subtraction before it
can learn anything else.

Configuration: Setup
-------------------------------------

.. code-block:: yaml

   feature_sets:
     - name: feature_set_1
       features:
         - neighbor_diff

Configuration: Parameters
-------------------------------------

.. code-block:: yaml

   feature_param_sets:
     - name: feature_set_1_param_set_1
       params:
         - feature: neighbor_diff
           col_names: [ temp, psal ]
           outputs: [ diff, large_diff_frac ]
           params: { lags: [ 1, 2, 3, 4, 5 ], directions: [ up, down ] }
           stats_set: { type: raw }

================================ =========================================
Output                           Columns
================================ =========================================
``diff``                         ``{variable}_diff_{direction}_{lag}``:
                                 the level less its neighbour that many
                                 levels up (shallower) or down (deeper).
``large_diff_frac``              ``{variable}_w{window}_large_diff_frac``:
                                 the fraction of the window whose
                                 reference difference counts as large.
================================ =========================================

================================ ==================== ===================
Parameter                        Default              Meaning
================================ ==================== ===================
``lags``                         ``[1, 2, 3, 4, 5]``  Neighbour distances.
``directions``                   ``[up, down]``       Shallower, deeper,
                                                      or both.
``windows``                      ``[5, 11, 21, 41]``  Windows for the
                                                      fraction.
``reference_lag``                ``1``                Which difference the
                                                      fraction counts.
``reference_direction``          ``up``               From which side.
``large_diff_z``                 ``3.0``              Robust score above
                                                      which a difference
                                                      is large.
``large_diff_threshold``         none                 An absolute
                                                      magnitude instead.
================================ ==================== ===================

Why several lags
-------------------------------------

The two failures these catch look different at different distances. A single
bad level stands out sharply at lag 1 and fades as the lag grows, because the
comparison moves further from the level that is wrong. A shifted or drifting
sensor shows up as a step that every lag sees. Giving the model both lets it
tell them apart.

The cost is levels: lag 5 has no answer for the first five levels of a profile
in the ``up`` direction and the last five in the ``down`` direction.

What counts as large
-------------------------------------

This is the one number the feature proposal leaves open, so there are two ways
to say it.

By default the reference difference is standardised against the profile's own
median and median absolute deviation and compared with ``large_diff_z``. That
needs no knowledge of the variable's units and adapts to how noisy the
individual profile is. Its one limitation is that a profile whose differences
are all identical has no scale at all, and the fraction is null there rather
than claiming an infinite score.

``large_diff_threshold`` compares the magnitude directly instead, either as one
number for every variable or as a mapping:

.. code-block:: yaml

   - feature: neighbor_diff
     col_names: [ temp, psal ]
     outputs: [ large_diff_frac ]
     params:
       large_diff_threshold: { temp: 1.0, psal: 0.3 }

A mapping that does not cover every configured variable is an error rather
than a silent default.
