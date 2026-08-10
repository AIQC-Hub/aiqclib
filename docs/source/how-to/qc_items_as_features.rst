Using NRT QC Items as Training Features
=======================================

Every NRT QC item is an ordinary feature class registered under a
``qc_``-prefixed name, so the same tests the :doc:`nrt_qc` module runs can also
be computed inside the Dataset Preparation pipeline and handed to the model as
input features. Instead of only seeing the measurements, the model then also
sees what the automated real-time tests made of them.

This is optional. The default feature sets do not include any QC items, and a
model trained without them works exactly as before.

Why you might want them
-----------------------

The QC items encode domain knowledge that a model would otherwise have to
rediscover from the raw values — that a measurement far from its vertical
neighbours is suspect, that pressure should increase down a profile, that
density should not invert. Supplying the outcome of each test as a feature
lets the model learn *when to trust those rules* rather than learning the rules
themselves, which is generally the easier problem.

The cost is that each item adds one column per checked variable, and the items
are computed for every observation during preparation, so a long feature list
makes the ``extract`` step slower.

Adding a QC item to a feature set
---------------------------------

A QC item is configured like any other feature: name it in ``feature_sets``,
and give it an entry in ``feature_param_sets``.

.. code-block:: yaml

   feature_sets:
     - name: feature_set_1
       features:
         - location
         - day_of_year
         - basic_values
         - qc_spike            # QC items are listed alongside the others
         - qc_global_range
         - qc_impossible_date

   feature_param_sets:
     - name: feature_set_1_param_set_1
       params:
         - feature: location
           stats_set: { type: raw }
           col_names: [ longitude, latitude ]
         - feature: qc_spike
           col_names: [ temp, psal ]
         - feature: qc_global_range
           col_names: [ temp, psal ]
           params: { temp: { min: -2.5, max: 40.0 },
                     psal: { min: 2.0, max: 41.0 } }
         - feature: qc_impossible_date

The available item names are the ``qc_``-prefixed registry names:
``qc_impossible_date``, ``qc_impossible_location``, ``qc_global_range``,
``qc_regional_range``, ``qc_pressure_increasing``, ``qc_spike``,
``qc_gradient``, ``qc_digit_rollover``, ``qc_stuck_value``,
``qc_density_inversion`` and ``qc_temp_to_psal``. For what each one checks, see
:ref:`nrt-qc-items`; for their default thresholds, see
:doc:`../configuration/nrtqc`.

The three settings
------------------

Each entry accepts the same three keys, all optional:

``params``
   Per-item thresholds, overriding the built-in defaults **key by key**.

``col_names``
   The variables to produce flag columns for.

``fail_flag``
   The flag emitted when the test fails — ``4`` (bad) by default, ``3`` to
   record the failure as "probably bad" instead.

.. important::

   ``params`` merges with the defaults rather than replacing them, so it does
   **not** narrow the list of variables. Overriding only the ``temp`` range of
   ``qc_global_range`` still yields both ``temp_qc_global_range`` and
   ``psal_qc_global_range``, because ``psal`` remains in the merged defaults.
   Use ``col_names`` when you want a subset:

   .. code-block:: yaml

      # Both temp and psal columns are produced
      - feature: qc_global_range
        params: { temp: { min: -2.5, max: 40.0 } }

      # Only temp_qc_global_range is produced
      - feature: qc_global_range
        col_names: [ temp ]

Unlike the measurement features, QC items need no ``stats_set``: their output
is already a small integer flag, so there is nothing to normalize.

The columns you get
-------------------

Item flags follow the same naming as the NRT QC module: variable-specific
items produce ``{variable}_qc_{item}`` and profile-level items produce a single
``qc_{item}`` column. The example above adds five columns to each target's
feature table:

.. code-block:: text

   temp_qc_spike
   psal_qc_spike
   temp_qc_global_range
   psal_qc_global_range
   qc_impossible_date

Each is never null — ``1`` where the test passed, the item's ``fail_flag``
where it failed — so they need no imputation.

.. note::

   These are the model's *inputs*. They are unrelated to the QC flag columns
   that supply the **labels** (``pos_flag_values`` / ``neg_flag_values`` in
   ``target_sets``). Using ``qc_spike`` as a feature says nothing about which
   observations count as bad; that is still decided by the target's ``flag``
   column.

Two items that need care
------------------------

``qc_regional_range``
   Has no built-in defaults, so it must be given the region's bounds through
   ``params`` or it raises an error rather than silently passing everything.

``qc_temp_to_psal``
   Reads an aggregated temperature flag column — ``temp_nrt_flag`` by default —
   which is produced by the NRT QC module, not by the preparation pipeline.
   Unless that column is already present in your input file, adding this item
   as a feature fails with ``ColumnNotFoundError: unable to find column
   "temp_nrt_flag"``. Run :doc:`nrt_qc` first and prepare from its output, or
   point ``params.source_column`` at a flag column your input does have.

Next Steps
----------

*   :doc:`nrt_qc` — running the QC items as a standalone module, and what each
    item checks.
*   :doc:`../configuration/preparation` — the full ``feature_sets`` and
    ``feature_param_sets`` reference.
*   :doc:`shap_values` — checking whether the added features actually
    contribute to the model.
