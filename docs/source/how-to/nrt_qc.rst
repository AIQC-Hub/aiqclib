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

The available QC items, their default thresholds, and all configuration
sections are described in :doc:`../configuration/nrtqc`.

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
