QC Items
===========================

The ``qc_*`` features are the automated real-time QC tests for Argo and CTD
profiles (the RTQC tests of the NRT QC recommendation document). Each one
emits a flag per observation: ``1`` where the test passed and the item's fail
flag (``4`` by default) where it failed. The columns are never null.

The same classes serve two pipelines. The `nrtqclib`_ package runs them on
their own and combines their flags into a final NRT flag per variable; the
Dataset Preparation pipeline here computes them like any other feature, so a
model can see what the real-time tests made of each measurement. This page is
the reference for each item. For how to use them as training features, and
the pitfalls of doing so, see :doc:`../how-to/qc_items_as_features`.

Configuration: Setup
-------------------------------------

Name the items in ``feature_sets`` alongside the other features:

.. code-block:: yaml

   feature_sets:
     - name: feature_set_1
       features:
         - basic_values
         - qc_spike
         - qc_global_range
         - qc_impossible_date

Configuration: Parameters
-------------------------------------

Every key is optional, and none of the items takes a ``stats_set``: a flag is
already a small integer code, so there is nothing to normalize.

.. code-block:: yaml

   feature_param_sets:
     - name: feature_set_1_param_set_1
       params:
         - feature: qc_spike
           col_names: [ temp, psal ]
         - feature: qc_global_range
           col_names: [ temp ]
           params: { temp: { min: -2.0, max: 35.0 } }
           fail_flag: 3
         - feature: qc_impossible_date

``params``
   Thresholds and settings, merged with the item's defaults **key by key**.
   Overriding the ``temp`` range of ``qc_global_range`` leaves the ``psal``
   default in place, so it still yields a ``psal`` column.

``col_names``
   The variables to produce flag columns for. This, not ``params``, is how to
   restrict an item to a subset of variables.

``fail_flag``
   The flag emitted on failure: ``4`` (bad) by default, or ``3`` (probably
   bad).

Output columns
-------------------------------------

Variable-specific items produce ``{variable}_qc_{item}`` (for example
``temp_qc_spike``); variable-independent items produce a single
``qc_{item}`` column (for example ``qc_impossible_date``).

The items
-------------------------------------

The **Level** column says whether a flag varies within a profile
(observation) or takes one value for the whole profile (profile). Only the
profile-level items can be used directly in the profile-level pipeline; the
others need an ``agg`` such as ``fail_frac`` there (see
:doc:`../how-to/profile_level_pipeline`).

.. list-table::
   :header-rows: 1
   :widths: 24 8 14 54

   * - Feature
     - Test
     - Level
     - Fails when
   * - ``qc_impossible_date``
     - RTQC2
     - profile
     - The profile timestamp is null, its year is not after ``min_year``, or
       it lies in the future.
   * - ``qc_impossible_location``
     - RTQC3
     - profile
     - Latitude or longitude is missing or outside its bounds.
   * - ``qc_position_on_land``
     - RTQC4
     - profile
     - The sea floor depth at the position says it is not in the ocean.
   * - ``qc_global_range``
     - RTQC6
     - observation
     - A value lies outside the variable's ``[min, max]`` (bounds inclusive).
   * - ``qc_regional_range``
     - RTQC7
     - observation
     - As ``qc_global_range``, with the region's own bounds.
   * - ``qc_pressure_increasing``
     - RTQC8
     - observation
     - Pressure repeats the previous value, or falls below the running
       maximum of the pressures before it.
   * - ``qc_spike``
     - RTQC9
     - observation
     - ``|V2 - (V3 + V1)/2| - |(V3 - V1)/2|`` exceeds the depth-dependent
       threshold, where V2 is the tested value and V1, V3 its neighbours.
   * - ``qc_gradient``
     - RTQC11
     - observation
     - ``|V2 - (V3 + V1)/2|`` exceeds the depth-dependent threshold.
   * - ``qc_digit_rollover``
     - RTQC12
     - observation
     - The absolute difference from the previous value exceeds the
       variable's threshold.
   * - ``qc_stuck_value``
     - RTQC13
     - profile
     - Every non-null value of the variable in the profile is identical.
   * - ``qc_density_inversion``
     - RTQC14
     - observation
     - The potential density (sigma-0) of a consecutive pair of levels
       inverts by more than the threshold. Both levels fail.
   * - ``qc_temp_to_psal``
     - rule
     - observation
     - The aggregated temperature flag is 3 or 4; that flag is copied to
       salinity.

A missing value passes, since it cannot be checked: a missing measurement
passes the observation-level tests, a missing sea floor depth passes
``qc_position_on_land``, and a variable with no values passes
``qc_stuck_value``. Only the date and location items fail on a missing value.
Every item treats a floating-point NaN exactly like null, so input that writes
NaN for missing values (as ctddump does) needs no conversion. Observations without
the neighbours a test needs (the ends of a profile for ``qc_spike`` and
``qc_gradient``, the first observation for ``qc_digit_rollover``) pass.

Default parameters
-------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 26 40 34

   * - Feature
     - Default ``params``
     - Output columns
   * - ``qc_impossible_date``
     - ``min_year: 1950``,
       ``timestamp_column: profile_timestamp``
     - ``qc_impossible_date``
   * - ``qc_impossible_location``
     - ``lat_min: -90``, ``lat_max: 90``,
       ``lon_min: -180``, ``lon_max: 180``
     - ``qc_impossible_location``
   * - ``qc_position_on_land``
     - ``depth_column: bathymetry``, ``positive_depth: true``
     - ``qc_position_on_land``
   * - ``qc_global_range``
     - ``temp: {min: -2.5, max: 40.0}``,
       ``psal: {min: 2.0, max: 41.0}``
     - ``temp_qc_global_range``, ``psal_qc_global_range``
   * - ``qc_regional_range``
     - none (must be given)
     - ``{variable}_qc_regional_range``
   * - ``qc_pressure_increasing``
     - none
     - ``qc_pressure_increasing``
   * - ``qc_spike``
     - ``temp: {shallow: 6.0, deep: 2.0}``,
       ``psal: {shallow: 0.9, deep: 0.3}``,
       ``depth_threshold: 500``
     - ``temp_qc_spike``, ``psal_qc_spike``
   * - ``qc_gradient``
     - ``temp: {shallow: 9.0, deep: 3.0}``,
       ``psal: {shallow: 1.5, deep: 0.5}``,
       ``depth_threshold: 500``
     - ``temp_qc_gradient``, ``psal_qc_gradient``
   * - ``qc_digit_rollover``
     - ``temp: 10.0``, ``psal: 5.0``
     - ``temp_qc_digit_rollover``, ``psal_qc_digit_rollover``
   * - ``qc_stuck_value``
     - ``temp: {}``, ``psal: {}``, ``min_observations: 2``
     - ``temp_qc_stuck_value``, ``psal_qc_stuck_value``
   * - ``qc_density_inversion``
     - ``threshold: 0.03`` (kg/m³)
     - ``temp_qc_density_inversion``, ``psal_qc_density_inversion``
   * - ``qc_temp_to_psal``
     - ``source_column: temp_nrt_flag``, ``target_variable: psal``
     - ``psal_qc_temp_to_psal``

For ``qc_spike`` and ``qc_gradient``, ``shallow`` applies where the pressure
is less than ``depth_threshold`` (decibars) and ``deep`` elsewhere. The empty
dictionaries of ``qc_stuck_value`` only mark which variables are checked.

Which items work as training features
-------------------------------------

All twelve are registered features, but three depend on something the
preparation input may not have:

``qc_regional_range``
   Has no defaults. Without the region's bounds in ``params`` it raises an
   error rather than passing everything, so it needs one configuration per
   region:

   .. code-block:: yaml

      - feature: qc_regional_range
        params: { temp: { min: -2.0, max: 24.0 },
                  psal: { min: 0.0, max: 38.0 } }

``qc_position_on_land``
   Reads the sea floor depth from an input column (``bathymetry`` by
   default); it does not look up a bathymetry grid. A missing column is an
   error. ``positive_depth: false`` is for inputs where depth below sea level
   is negative. Sea level itself counts as land.

``qc_temp_to_psal``
   Reads the aggregated temperature flag ``temp_nrt_flag``, which the
   `nrtqclib`_ pipeline writes. Prepare from its output, or point
   ``source_column`` at a flag column the input does have. Its propagated
   flag keeps its own severity (3 or 4), so ``fail_flag`` does not apply.

The other nine run on any input with the usual profile columns. Before relying
on them, check the points in :doc:`../how-to/qc_items_as_features`: labels
taken from NRT flags make the items partly circular, many items never fire on
a given dataset (a constant column), and the two
``qc_density_inversion`` columns are always identical.

.. _nrtqclib: https://github.com/AIQC-Hub/nrtqclib
