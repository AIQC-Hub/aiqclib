Running Near-Real Time Quality Control (NRT QC)
===============================================

The NRT QC module applies the automated real-time QC tests recommended for
temperature and salinity profiles (Argo/CTD RTQC tests) to an input dataset
and writes the original parquet enriched with:

* **one flag column per QC item** — e.g. ``temp_qc_spike``,
  ``qc_impossible_date`` — usable directly as training features, and
* **a final NRT flag per variable** — ``temp_nrt_flag`` / ``psal_nrt_flag``,
  the most severe flag among the variable's applicable item columns.

Flags follow the IOC/Argo scheme: 1 (good), 3 (probably bad), 4 (bad).

Quick start
-----------

.. code-block:: python

   import aiqclib as aq

   aq.write_config_template(file_name="/path/to/nrt_qc_config.yaml", stage="nrt_qc")
   # Edit the paths, variables, and QC items in the YAML, then:
   config = aq.read_config("/path/to/nrt_qc_config.yaml")
   aq.run_nrt_qc(config)

The QC items
------------

``aiqclib`` provides eleven QC items. Ten implement the Argo/CTD real-time
tests (RTQC2 to RTQC14); ``temp_to_psal`` is the propagation rule from the
NRT QC recommendation document rather than a numbered test. An item runs
only when it is listed in the ``qc_item_sets`` section of the configuration
file.

Profile-level tests produce one result for the whole profile, applied to
every observation in it:

``impossible_date`` (RTQC2)
   The profile timestamp must be present, later than ``min_year`` (1950 by
   default), and not in the future at processing time.

``impossible_location`` (RTQC3)
   Latitude must lie within [-90, 90] and longitude within [-180, 180]. A
   missing position fails.

``stuck_value`` (RTQC13)
   All non-null measurements of a variable being identical indicates a stuck
   sensor, so the variable is flagged throughout the profile. Profiles with
   fewer than ``min_observations`` (2 by default) measurements are exempt, as
   are variables that were not measured.

Observation-level tests produce one result per measurement:

``global_range`` (RTQC6)
   A gross filter wide enough to accommodate all expected ocean extremes:
   temperature within [-2.5, 40.0] °C and salinity within [2.0, 41.0] by
   default. Null values pass, since a missing measurement cannot be
   range-checked.

``regional_range`` (RTQC7)
   The same check against the tighter ranges of the configuration file's
   region. There are **no built-in defaults** — supply your region's bounds
   or the item raises an error rather than silently passing everything.

``pressure_increasing`` (RTQC8)
   Pressures must increase down the profile. Within a run of constant
   pressure all but the first observation are flagged; where pressure
   reverses, every observation below the running maximum is flagged. Pressure
   is shared by all variables, so this test yields a single column.

``spike`` (RTQC9)
   A measurement that differs sharply from both vertical neighbours in size
   and gradient, scored as ``|V2 - (V3 + V1)/2| - |(V3 - V1)/2|``. Thresholds
   are depth-dependent (temperature 6.0/2.0 °C and salinity 0.9/0.3 either
   side of 500 db by default). Observations at a profile boundary always pass.

``gradient`` (RTQC11)
   Too steep a difference from the vertical neighbours, scored as
   ``|V2 - (V3 + V1)/2|`` against depth-dependent thresholds (temperature
   9.0/3.0 °C and salinity 1.5/0.5 by default).

``digit_rollover`` (RTQC12)
   Sensors store values in a limited bit range and wrap around to the low end
   when it is exceeded. An uncompensated rollover shows up as a jump from the
   previous observation beyond the threshold (10.0 °C for temperature, 5.0
   for salinity by default).

``density_inversion`` (RTQC14)
   The potential density anomaly sigma-0 (UNESCO 1983) must not decrease with
   depth by more than ``threshold`` (0.03 kg/m³ by default). Consecutive
   levels are compared in both directions, so both members of an inverted
   pair are flagged. Density combines temperature and salinity, so the two
   variables are flagged jointly.

``temp_to_psal``
   Not a test but a propagation rule: where the aggregated temperature flag
   is 3 or 4, salinity inherits it at the same severity. See
   `Temperature-to-salinity propagation`_ below.

For the output column names, the full default parameters, and the YAML
syntax for overriding them, see :doc:`../configuration/nrtqc`.

One configuration file per region
---------------------------------

Some tests are region dependent (most notably the **regional range** test).
Instead of detecting the region from coordinates, prepare **one configuration
file per region** (e.g. Arctic, Baltic, Mediterranean): the files share the
same structure and differ only in region-dependent parameters such as the
regional ranges or the density inversion threshold.

Temperature-to-salinity propagation
-----------------------------------

When salinity is computed from temperature and conductivity, a temperature
flagged 4 (or 3) corrupts the salinity too. Enabling the ``temp_to_psal``
item propagates the final temperature flag onto salinity with its severity,
recorded in its own ``psal_qc_temp_to_psal`` column so the propagation stays
traceable. Datasets with independently measured salinity simply omit the item.

Comparing against existing flags
--------------------------------

When the input already carries NRT QC flags (e.g. ``temp_qc``, ``psal_qc``),
give each variable its ``flag`` column in the ``qc_variable_sets`` section:

.. code-block:: yaml

   qc_variable_sets:
     - name: qc_variable_set_1
       variables:
         - name: temp
           flag: temp_qc
           pos_flag_values: [ 4, 6, 7 ]  # optional: for agreement metrics
           neg_flag_values: [ 1 ]

``run_nrt_qc`` then writes one comparison report per variable
(``nrt_qc_flag_comparison_{variable}.tsv``) containing a contingency table
of existing vs new flag values, binary agreement metrics (accuracy,
precision, recall — only when the pos/neg flag values are given), and a
per-item breakdown showing which items drive the disagreements. Variables
without a ``flag`` are skipped; omit all flags to skip the comparison step
entirely.

The existing flag column may hold integers, strings or floats, and the
configured values may be written as ``4`` or as ``"4"`` (see
:ref:`qc-flag-columns`). Existing flags that are missing or unreadable get
their own row in the contingency table, so they stay visible instead of being
folded into a real flag value; the agreement metrics count only the values
listed in ``pos_flag_values`` / ``neg_flag_values``. The input's flag columns
are written to the output unchanged, whatever type they use.

Using QC items as training features
-----------------------------------

Every QC item is an ordinary feature class registered under a
``qc_``-prefixed name (``qc_spike``, ``qc_global_range``, …), so the flag
columns can also be produced inside the Dataset Preparation pipeline by
adding the names to a ``feature_set``.
