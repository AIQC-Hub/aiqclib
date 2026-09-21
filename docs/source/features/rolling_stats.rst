Window Statistics
===========================

The ``rolling_stats`` feature is an observation-level feature describing the
window of levels around each measurement, at one or more widths.

Where :doc:`profile_smooth` asks whether a measurement disagrees with its
neighbours, this asks what the neighbourhood itself looks like: how warm, how
variable, how tightly packed the levels around this one are. That is the
context those anomaly scores are judged against.

Configuration: Setup
-------------------------------------

.. code-block:: yaml

   feature_sets:
     - name: feature_set_1
       features:
         - rolling_stats

Configuration: Parameters
-------------------------------------

.. code-block:: yaml

   feature_param_sets:
     - name: feature_set_1_param_set_1
       params:
         - feature: rolling_stats
           col_names: [ temp, psal ]
           outputs: [ median, mad, robust_z ]
           params: { windows: [ 5, 11, 21, 41 ] }
           stats_set: { type: raw }

Every output is produced per variable and per window, named
``{variable}_w{window}_{output}``.

================================ =========================================
Output                           What it holds
================================ =========================================
``mean``, ``median``             The centre of the window.
``min``, ``max``                 Its extremes.
``std``                          Its standard deviation.
``mad``                          Its median absolute deviation: the median
                                 of the deviations from the window's own
                                 median.
``robust_z``                     The level's own value measured against
                                 that median and deviation. This is the
                                 proposal's "local robust z-score".
================================ =========================================

================================ ==================== ===================
Parameter                        Default              Meaning
================================ ==================== ===================
``windows``                      ``[5, 11, 21, 41]``  Window sizes, odd.
``min_samples``                  none                 Non-null values a
                                                      window needs. The
                                                      default is the full
                                                      window.
================================ ==================== ===================

Prefer the robust pair
-------------------------------------

``mad`` and ``robust_z`` are the ones to reach for rather than ``std`` and a
plain z-score. A single bad level moves a window's mean and inflates its
standard deviation, so the very thing being looked for corrupts the yardstick
used to look for it. A median and a median absolute deviation barely notice it.

``robust_z`` is null where the window has no spread at all. A window of
identical values offers no scale to be unusual against, and calling the
departure infinite would be a claim the data does not support.

Why several widths
-------------------------------------

A narrow window describes the measurement's immediate surroundings and a wide
one the layer it sits in. A measurement that disagrees with the first but not
the second is a different kind of suspect from one that disagrees with both:
the first looks like a single bad level, the second like a displaced or
drifting sensor.

Edges and profile length
-------------------------------------

A window of ``w`` points needs ``(w - 1) / 2`` levels on each side, so the
outputs are null near the ends of a profile, and a window wider than the
profile produces nothing at all. ``min_samples`` relaxes that, at the cost of
statistics computed from a partial window near the surface and the sea floor.
Check the typical number of levels in your profiles before adding a 41 point
window.

Column count
-------------------------------------

Two variables, four windows and all seven outputs is 56 columns from this one
entry. The ``outputs`` list is how that is kept in hand; a good starting subset
is ``median``, ``mad`` and ``robust_z`` at two widths.
