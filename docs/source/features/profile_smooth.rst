Profile Smoothing
===========================

The ``profile_smooth`` feature is an observation-level feature that fits a
smooth curve through each profile and describes how each measurement relates
to it.

A profile is a physical object, so a smooth curve through it is a fair guess at
what the water column is doing. What a measurement does that the curve does not
is the part worth flagging. That gives three families of column: the curve and
its slope and curvature, the residual and its robust score, and the two numbers
that tell a spike from a real feature.

Configuration: Setup
-------------------------------------

.. code-block:: yaml

   feature_sets:
     - name: feature_set_1
       features:
         - profile_smooth

Configuration: Parameters
-------------------------------------

.. code-block:: yaml

   feature_param_sets:
     - name: feature_set_1_param_set_1
       params:
         - feature: profile_smooth
           col_names: [ temp, psal ]
           outputs: [ residual, robust_z, spike_index, curvature_ratio ]
           params: { window: 11, polyorder: 2, windows: [ 11, 41 ] }
           stats_set: { type: raw }

``col_names`` names the variables to smooth. Per-level outputs are named
``{variable}_{output}`` and windowed ones ``{variable}_w{window}_{output}``.

================================ =========================================
Output                           What it holds
================================ =========================================
``smooth``                       The fitted value at the level.
``d1``, ``d2``                   Slope and curvature of the fit, per unit
                                 of ``spacing_column``.
``residual``                     The measurement less the fit.
``robust_z``                     The residual standardised by the
                                 profile's own median and median absolute
                                 deviation, so it is comparable between
                                 profiles and variables without any unit
                                 knowledge.
``curvature_ratio``              ``|d2| / |residual|``.
``spike_index``                  The Argo RTQC9 stencil value,
                                 unthresholded.
``outlier_frac``                 Per window: the fraction whose
                                 ``robust_z`` exceeds ``outlier_z``.
``high_curvature_frac``          Per window: the same for the curvature.
================================ =========================================

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Parameter
     - Default
     - Meaning
   * - ``window``
     - ``11``
     - Points in the smoothing window (odd).
   * - ``polyorder``
     - ``2``
     - Degree of the fitted polynomial.
   * - ``spacing_column``
     - ``pres``
     - Derivatives are per unit of this column; ``null`` keeps them per level.
   * - ``min_spacing``
     - ``1e-6``
     - Below this step the derivative conversion is null.
   * - ``windows``
     - ``[5, 11, 21, 41]``
     - Windows for the fraction outputs.
   * - ``outlier_z``
     - ``3.0``
     - What counts as an outlying residual.
   * - ``high_curvature_z``
     - ``3.0``
     - What counts as high curvature.
   * - ``ratio_floor``
     - ``1e-9``
     - Residual magnitude below which ``curvature_ratio`` is null.

Telling a spike from a thermocline
-------------------------------------

A large residual alone cannot distinguish the two cases that matter most: a
measurement that jumps away from a smooth column and comes straight back, and a
measurement sitting on a real, sharp feature. The pair ``curvature_ratio`` and
``spike_index`` can.

A sharp but real feature bends the fitted curve, so it lands in ``d2`` and the
ratio is high. A spike leaves the curve alone, so it lands in ``residual`` and
the ratio is low. ``spike_index`` says the same thing from the other side: it
measures the level against the average of its two neighbours and then subtracts
the size of the gradient across them, so a steady ramp scores negative however
steep it is, and only a level that departs from both neighbours in the same
direction scores positive.

Why Savitzky-Golay
-------------------------------------

A moving average flattens the peak it is passing over, which is the opposite of
what is wanted when the peak is the thing being measured. A Savitzky-Golay
filter fits a polynomial instead, so it preserves the height and width of a
feature while removing noise, and it yields the derivatives for free. It is a
fixed set of weights, so it needs no signal processing dependency.

Choosing a window
-------------------------------------

A window of ``w`` points needs ``(w - 1) / 2`` levels on each side, and the
profile ends are null. On a 20 level profile a 41 point window produces nothing
at all. Start with a window well under half the typical profile length, and use
a second, wider entry rather than one compromise window when both scales
matter: two entries of the same feature are legal, and the window appears in
the windowed column names.

Normalization
-------------------------------------

``robust_z``, ``curvature_ratio`` and the fractions are already comparable
across datasets, so ``raw`` is the right stats set for them. The columns in the
variable's own units (``smooth``, ``d1``, ``d2``, ``residual``,
``spike_index``) can be given an explicit ``min_max`` stats set; data-derived
normalization is not available for them, because it is fitted from the step 2
summary table, which has rows only for the raw input variables.
