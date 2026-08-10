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

Potential issues
----------------

These features behave differently from the measurement features, and the
differences are easy to miss because nothing fails loudly.

Circularity when the labels come from NRT flags
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is the one to think about first.

The QC items implement the Argo/CTD real-time tests. The ``temp_qc`` /
``psal_qc`` columns that these tutorials label from are **near-real-time QC
flags** (see :ref:`which-qc-flags`) — which the data provider produced by
running largely those same tests. ``aiqclib`` treats the two as measuring the
same thing elsewhere: the NRT QC module's comparison step scores its computed
items against the input's existing flags with accuracy, precision and recall.

So if you label from an NRT flag column *and* feed in the QC items as
features, you are partly giving the model the recipe that produced its own
labels. It can score well by rediscovering the provider's rules while learning
nothing about whether an observation is genuinely bad, and that skill will not
transfer to delayed-mode or expert-reviewed labels.

This does not make the combination wrong, but it changes what a good score
means. It is most defensible when the labels come from a source *independent*
of the automated tests — delayed-mode flags, or expert review. When labelling
from NRT flags, treat a jump in performance after adding QC items as a
suspicious result to investigate, not a win.

Items that never fire
^^^^^^^^^^^^^^^^^^^^^

Most QC tests fail rarely, and a feature with a single distinct value carries
no information — it only widens the table and slows training. Running six items
over the test fixtures produced ten columns, of which eight were constant:

.. code-block:: text

   temp_qc_spike               distinct=[1]
   psal_qc_spike               distinct=[1]
   temp_qc_gradient            distinct=[1]
   psal_qc_gradient            distinct=[1]
   temp_qc_density_inversion   distinct=[1, 4]
   psal_qc_density_inversion   distinct=[1, 4]
   qc_impossible_date          distinct=[1]
   temp_qc_stuck_value         distinct=[1]
   psal_qc_stuck_value         distinct=[1]
   qc_pressure_increasing      distinct=[1]

Check which items actually fire on *your* data before committing to a feature
set, and drop the ones that never do:

.. code-block:: python

   df.select(pl.col("^.*qc_.*$").n_unique())

Duplicated and overlapping columns
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``qc_density_inversion`` flags temperature and salinity **jointly**, so
``temp_qc_density_inversion`` and ``psal_qc_density_inversion`` are always
identical — two perfectly collinear columns. Keep one.

Others overlap without being identical: ``qc_spike`` and ``qc_gradient`` differ
only in their test statistic and fire on similar observations. That is harmless
for tree-based models but matters for the linear ones (Logistic Regression,
LDA, SVM), where collinear inputs make coefficients unstable and hard to read.

Profile-level items (``qc_impossible_date``, ``qc_impossible_location``,
``qc_stuck_value``) take one value for every observation in a profile, so they
carry far less per-row information than their column count suggests.

Flag values are codes, not magnitudes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A flag column holds ``1`` for pass and the ``fail_flag`` (``4``, or ``3``) for
fail. Those are category codes, but they arrive as numbers, and the gap between
them is arbitrary — nothing means a failure is "three units worse" than a pass.
Tree-based models are unaffected, since they only split on the value. Distance-
and coefficient-based models (KNN, SVM, Logistic Regression, LDA, MLP) do read
the magnitude, so switching an item's ``fail_flag`` from ``4`` to ``3`` changes
its influence for those algorithms even though the pass/fail meaning is
unchanged.

Items that need extra configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``qc_regional_range``
   Has no built-in defaults, so it must be given the region's bounds through
   ``params`` or it raises an error rather than silently passing everything.
   As in the NRT QC module, this means one configuration per region.

``qc_temp_to_psal``
   Reads an aggregated temperature flag column — ``temp_nrt_flag`` by default —
   which is produced by the NRT QC module, not by the preparation pipeline.
   Unless that column is already present in your input file, adding this item
   as a feature fails with ``ColumnNotFoundError: unable to find column
   "temp_nrt_flag"``. Run :doc:`nrt_qc` first and prepare from its output, or
   point ``params.source_column`` at a flag column your input does have.

Cost
^^^^

Every listed item is computed for each observation during ``extract``, and the
vertical tests (``qc_spike``, ``qc_gradient``, ``qc_density_inversion``) need
neighbouring rows. A long item list makes preparation slower and the feature
table wider without necessarily making the model better — :doc:`shap_values`
is the way to check whether the added columns are earning their place.

Next Steps
----------

*   :doc:`nrt_qc` — running the QC items as a standalone module, and what each
    item checks.
*   :doc:`../configuration/preparation` — the full ``feature_sets`` and
    ``feature_param_sets`` reference.
*   :doc:`shap_values` — checking whether the added features actually
    contribute to the model.
