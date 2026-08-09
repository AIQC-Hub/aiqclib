Classifying With or Without Labels
==================================

The classification pipeline can run in two modes, chosen per target from
whether a QC ``flag`` column is available:

* **With labels** — a ``flag`` is given, so ground-truth labels are built and
  the model's performance is evaluated.
* **Without labels** — no ``flag`` is available (unlabeled data), so the
  pipeline predicts every row but skips evaluation.

You do not need a dummy QC column for unlabeled data.

Option 1: with labels (default)
-------------------------------

Give each target its QC ``flag`` and the flag values that mark positive and
negative rows:

.. code-block:: yaml

   target_sets:
     - name: target_set_1
       variables:
         - name: temp
           flag: temp_qc
           pos_flag_values: [ 3, 4 ]
           neg_flag_values: [ 1, 2 ]

Rows are labelled from the flag, and the classify phase writes predictions plus
the evaluation artefacts — the report, the **model-scores** file
(see :doc:`performance_evaluation`), and the metric plots.

The flag column may hold integers, strings or floats, and the values may be
written as ``4`` or as ``"4"`` (see :ref:`qc-flag-columns`). Rows whose flag is
missing or unreadable match neither list and are dropped from the labelled set,
exactly like a flag value listed in neither ``pos_flag_values`` nor
``neg_flag_values``.

Option 2: without labels
------------------------

Omit ``flag`` (or set it to ``null`` / empty) for a target. The pipeline then
keeps **all** rows, emits null ``flag``/``label`` columns, and skips label
creation and performance evaluation:

.. code-block:: yaml

   target_sets:
     - name: target_set_1
       variables:
         - name: temp        # no `flag` — classify without evaluation

Predictions (``predicted_label`` and ``score``) are still produced; only the
report, model-scores, and metric-plot files are skipped.

To force this for every target in a step regardless of ``flag``, set the
step-level override under the ``model`` step parameters:

.. code-block:: yaml

   step_param_set:
     steps:
       model:
         skip_evaluation: true   # omit to auto-detect per target from `flag`

.. note::

   Only an *absent or empty* ``flag`` triggers the label-free path. A ``flag``
   that names a column missing from the data is still treated as an error, so
   typos are not silently ignored.

SHAP values
-----------

SHAP explains predictions, not labels, so it is available in **both** modes and
is controlled independently by ``calculate_shap`` (see :doc:`shap_values`):

.. code-block:: yaml

   step_param_set:
     steps:
       model:
         calculate_shap: true

In the label-free mode the ``label`` column of the SHAP output is null, while
the per-feature SHAP contributions and the ``score`` are unaffected.
