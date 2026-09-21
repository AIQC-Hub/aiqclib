Regime Flags
===========================

The ``regime_flags`` feature is an observation-level feature saying where in
the water column a measurement sits: in the mixed layer, in the steepest part
of a variable's gradient, and how far it is from that steepest part.

The same measurement means different things in different places. A half degree
step between two levels is ordinary inside a thermocline and alarming inside a
mixed layer, where by definition the water is uniform. The anomaly scores
computed from neighbouring levels cannot know which of those they are looking
at. These columns tell them.

Configuration: Setup
-------------------------------------

.. code-block:: yaml

   feature_sets:
     - name: feature_set_1
       features:
         - regime_flags

Configuration: Parameters
-------------------------------------

.. code-block:: yaml

   feature_param_sets:
     - name: feature_set_1_param_set_1
       params:
         - feature: regime_flags
           col_names: [ temp, psal ]
           params: { mixed_layer_criterion: density, density_threshold: 0.03 }
           stats_set: { type: raw }

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Output
     - Columns
   * - ``in_mixed_layer``
     - ``in_mixed_layer``: 1 above the mixed layer depth, 0 below it. This
       one is not per variable, since there is one mixed layer per profile.
   * - ``in_gradient_layer``
     - ``{variable}_in_gradient_layer``: 1 in the steepest part of that
       variable's gradient. For temperature this is the thermocline and for
       salinity the halocline; the name is general because the test is.
   * - ``normalized_depth_to_peak_gradient``
     - ``{variable}_normalized_depth_to_peak_gradient``: the level's
       pressure less the pressure of the steepest gradient, divided by the
       profile's pressure range. It is 0 at the peak, negative above and
       positive below.


================================ ================== =====================
Parameter                        Default            Meaning
================================ ================== =====================
``mixed_layer_criterion``        ``density``        Or ``temperature``.
``density_threshold``            ``0.03``           kg/m³ departure that
                                                    ends the mixed layer.
``temperature_threshold``        ``0.2``            °C departure, for the
                                                    temperature criterion.
``reference_pressure``           ``10.0``           The decibar level the
                                                    criterion measures
                                                    from, chosen to sit
                                                    below the diurnal
                                                    surface layer.
``gradient_percentile``          ``0.9``            The quantile of
                                                    gradient magnitude
                                                    that counts as steep.
``min_pressure_separation``      ``0.01``           Guards the gradient
                                                    division.
================================ ================== =====================

Why a position and not only a flag
-------------------------------------

``in_gradient_layer`` answers in or out.
``normalized_depth_to_peak_gradient`` answers where, on a scale that means the
same thing on a 50 dbar shelf cast and a 2000 dbar deep cast. That lets a model
learn "just below the thermocline", which is where a sensor settling after a
rapid temperature change tends to misbehave, and which a flag cannot express.

Two judgement calls
-------------------------------------

The definitions here are choices rather than standards, so two of them are
worth stating plainly.

A profile that never crosses the mixed layer criterion is mixed all the way
down. Every level gets 1, not null: that is an answer, not a missing value.

``in_gradient_layer`` additionally requires a non-zero gradient. A profile that
is uniform over most of its length has a gradient percentile of zero, and "at
or above zero" would otherwise flag the flat part as the steepest part of the
water column.
