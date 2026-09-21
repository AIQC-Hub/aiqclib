Building a Feature Set from the Profile Shape
==============================================

``aiqclib`` ships seven feature groups that read a whole profile rather than a
single observation. This guide is about choosing among them: what each one
answers, how they fit together, and how to keep the resulting training frame a
sensible size.

Each group has its own reference page with the full parameter table; the links
are in the table below.

What each group answers
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Group
     - The question it answers
   * - :doc:`../features/geo_context`
     - What kind of place is this, and where in the water column is this
       level?
   * - :doc:`../features/derived_values`
     - What are the quantities that follow from the measurements: density,
       depth, potential temperature?
   * - :doc:`../features/stratification`
     - Could this water column exist? Is the water below denser than the
       water above?
   * - :doc:`../features/profile_smooth`
     - Does this measurement disagree with a smooth curve through its
       neighbours, and is that a spike or a real feature?
   * - :doc:`../features/neighbor_diff`
     - How far is this measurement from the ones above and below it?
   * - :doc:`../features/rolling_stats`
     - What does the neighbourhood itself look like, at several widths?
   * - :doc:`../features/regime_flags`
     - Is this level in a mixed layer, or in the steepest part of the
       gradient?

The three kinds of evidence
---------------------------

The groups divide into three kinds, and a model does better with some of each
than with many columns of one.

**Physical constraints.** ``stratification`` says whether the water column is
possible. This is the only evidence that does not depend on the rest of the
dataset: an inverted column is wrong whether or not anything like it has been
seen before.

**Statistical anomaly.** ``profile_smooth``, ``neighbor_diff`` and
``rolling_stats`` say whether a measurement is unusual compared with its
surroundings. These are the anomaly detection core, and they are where most of
the columns come from.

**Context.** ``geo_context``, ``regime_flags`` and ``derived_values`` say what
kind of place and what kind of water this is. On their own they flag nothing;
their job is to let the model apply a different standard in a mixed layer than
in a thermocline, and on a shelf than in a basin.

A worked configuration
----------------------

A starting point that exercises all three kinds without producing hundreds of
columns:

.. code-block:: yaml

   feature_sets:
     - name: feature_set_1
       features:
         - location
         - day_of_year
         - basic_values
         - derived_values
         - stratification
         - profile_smooth
         - neighbor_diff
         - rolling_stats
         - regime_flags

   feature_param_sets:
     - name: feature_set_1_param_set_1
       params:
         - feature: location
           stats_set: { type: raw }
           col_names: [ longitude, latitude ]
         - feature: day_of_year
           convert: cosine
           col_names: [ profile_timestamp ]
         - feature: basic_values
           stats_set: { type: raw }
           col_names: [ temp, psal, pres ]

         - feature: derived_values
           outputs: [ sigma0, depth ]
           stats_set: { type: raw }

         - feature: stratification
           outputs: [ n2, unstable_flag ]
           stats_set: { type: raw }

         - feature: profile_smooth
           col_names: [ temp, psal ]
           outputs: [ residual, robust_z, spike_index, curvature_ratio,
                      outlier_frac ]
           params: { window: 11, polyorder: 2, windows: [ 11 ] }
           stats_set: { type: raw }

         - feature: neighbor_diff
           col_names: [ temp, psal ]
           outputs: [ diff ]
           params: { lags: [ 1, 2, 3 ], directions: [ up, down ] }
           stats_set: { type: raw }

         - feature: rolling_stats
           col_names: [ temp, psal ]
           outputs: [ median, mad, robust_z ]
           params: { windows: [ 5, 21 ] }
           stats_set: { type: raw }

         - feature: regime_flags
           col_names: [ temp, psal ]
           stats_set: { type: raw }

Add ``geo_context`` when the input carries ``bathymetry``:

.. code-block:: yaml

         - feature: geo_context
           outputs: [ bathymetry, normalized_depth, distance_to_bottom ]
           stats_set: { type: raw }

Keeping the frame a sensible size
---------------------------------

These groups can widen a training frame quickly. ``rolling_stats`` alone, with
two variables, four windows and every output, is 56 columns.

Three things keep that in hand.

``outputs`` names the columns to emit, and omitting it means all of them, so
always name them once a group is doing what you want. ``windows`` and ``lags``
multiply, so trim them before trimming anything else: two window widths usually
carry nearly as much as four. And a group can be listed twice with different
parameters, which is better than one compromise setting, because the window
size appears in the column name and the two entries do not collide.

Choosing window sizes
---------------------

Every windowed output needs ``(w - 1) / 2`` levels on each side of the level it
describes, and profile ends are null. A 41 point window on a 20 level profile
produces nothing at all.

Look at the distribution of profile lengths in your input before choosing.
A good rule is that the widest window should be well under half the typical
profile length, and the narrowest should be 5 or so, which is enough for a
median to mean something.

Normalization
-------------

Data-derived normalization (``auto_min_max``, ``standard``) is fitted from the
step 2 summary table, which has rows only for the raw input variables. The
columns these groups produce are derived quantities, so they are not covered.

In practice this matters less than it sounds, because the outputs worth
reaching for first are already comparable across datasets: every ``robust_z``,
every fraction and every flag is scale free by construction. Use ``raw`` for
those. For the columns in a variable's own units, supply an explicit
``min_max`` stats set. See :doc:`feature_normalization`.

Using these at profile level
----------------------------

All seven groups are observation level, so in the profile-level pipeline they
need an ``agg`` list saying how to aggregate each one over the profile:

.. code-block:: yaml

         - feature: profile_smooth
           col_names: [ temp, psal ]
           outputs: [ robust_z, spike_index ]
           agg: [ max, mean ]
           stats_set: { type: raw }

Using one without ``agg`` there is an error rather than a silent mismatch.
``max`` and ``fail_frac`` are usually the informative aggregations for an
anomaly score: whether the profile contains anything unusual, and how much of
it. See :doc:`profile_level_pipeline`.

Missing values and profile edges
--------------------------------

Every one of these features emits null rather than an estimate where it cannot
answer: at profile edges, where a window runs past the end of a profile, where
an input is missing, and where a value is outside the domain of the equation of
state. Null is what the models already treat as missing, and it cannot be
mistaken for a measurement.

This means the first and last few levels of every profile carry fewer features
than the middle. That is honest rather than convenient, and it is worth
remembering when a model appears to behave differently near the surface.
